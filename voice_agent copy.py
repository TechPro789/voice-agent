import os
import asyncio
import logging
from typing import AsyncIterable
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession, ModelSettings
from livekit.plugins import openai, sarvam

load_dotenv()

logger = logging.getLogger("ai-voice-agent")
logger.setLevel(logging.INFO)

# All Indic scripts supported by Sarvam bulbul:v2
SCRIPT_RANGES = {
    "bn-IN": (0x0980, 0x09FF),   # Bengali
    "hi-IN": (0x0900, 0x097F),   # Hindi (Devanagari) — also Marathi/Nepali
    "te-IN": (0x0C00, 0x0C7F),   # Telugu
    "ta-IN": (0x0B80, 0x0BFF),   # Tamil
    "kn-IN": (0x0C80, 0x0CFF),   # Kannada
    "ml-IN": (0x0D00, 0x0D7F),   # Malayalam
    "gu-IN": (0x0A80, 0x0AFF),   # ✅ Gujarati (was missing!)
    "pa-IN": (0x0A00, 0x0A7F),   # Punjabi (Gurmukhi)
    "or-IN": (0x0B00, 0x0B7F),   # Odia
}

def detect_language(text: str) -> str:
    """Detect language from Unicode script ranges."""
    counts = {lang: 0 for lang in SCRIPT_RANGES}
    for char in text:
        cp = ord(char)
        for lang, (start, end) in SCRIPT_RANGES.items():
            if start <= cp <= end:
                counts[lang] += 1
    best = max(counts, key=counts.get)
    if counts[best] > 2:
        logger.info(f"🌐 Detected: {best} ({counts[best]} chars)")
        return best
    return "en-IN"


INSTRUCTIONS = """
You are Mia from ABC Games — a friendly voice agent calling to reactivate users.

## CRITICAL LANGUAGE RULES:
- Detect the language from the customer's message and ALWAYS reply in that SAME language
- Gujarati spoken → reply ONLY in Gujarati
- Hindi spoken → reply ONLY in Hindi  
- Bengali spoken → reply ONLY in Bengali
- Telugu spoken → reply ONLY in Telugu
- English spoken → reply ONLY in English
- NEVER mix languages in a single response
- NEVER switch language on your own — only follow the customer

## PITCH ORDER (follow this across the conversation):
1. Warm greeting + introduce as Mia from ABC Games
2. Note they haven't visited recently
3. Offer: 100% deposit match up to ₹500
4. Bonus: Free spins included
5. Urgency: Offer valid for 3 days only
6. Polite close

## STYLE:
- Max 1-2 short sentences per reply
- Warm, natural — never robotic
- Persuasive but not pushy
"""


class AIVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self._current_lang = "en-IN"

    async def on_enter(self) -> None:
        logger.info("✅ on_enter — generating greeting")
        await self.session.generate_reply(
            instructions="Give a single short warm sentence greeting. Introduce yourself as Mia from ABC Games. English only."
        )

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ):
        # Buffer full text to detect language
        chunks = []
        async for chunk in text:
            chunks.append(chunk)
        full_text = "".join(chunks).strip()

        if not full_text:
            return

        detected = detect_language(full_text)
        if detected != self._current_lang:
            logger.info(f"🔄 TTS: {self._current_lang} → {detected}")
            self._current_lang = detected

        logger.info(f"🗣️ [{self._current_lang}]: {full_text[:80]}")

        # ✅ Sarvam is native WebSocket streaming — no StreamAdapter needed
        tts_instance = sarvam.TTS(
            target_language_code=self._current_lang,
            model="bulbul:v2",
            speaker="anushka",
            min_buffer_size=30,
        )

        async with tts_instance.stream() as stream:
            stream.push_text(full_text)
            stream.end_input()
            async for ev in stream:
                yield ev.frame


async def entrypoint(ctx: JobContext):
    logger.info(f"User joined room: {ctx.room.name}")

    if not os.getenv("SARVAM_API_KEY"):
        logger.error("❌ SARVAM_API_KEY missing!"); return
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("❌ OPENAI_API_KEY missing!"); return

    logger.info("✅ All API keys loaded")
    await ctx.connect()

    session = AgentSession(
        stt=sarvam.STT(
            language="unknown",
            model="saarika:v2.5",
            mode="transcribe",
        ),
        llm=openai.LLM(
            model="gpt-4o",
            temperature=0.7,
        ),
        tts=sarvam.TTS(
            target_language_code="en-IN",
            model="bulbul:v2",
            speaker="anushka",
            min_buffer_size=30,
        ),
    )

    await session.start(agent=AIVoiceAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))