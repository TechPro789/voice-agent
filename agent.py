import os
import asyncio
import logging
from typing import AsyncIterable
from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession, ModelSettings
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import openai, sarvam, silero
from qdrant_client import QdrantClient

load_dotenv()

logger = logging.getLogger("ai-voice-agent")
logger.setLevel(logging.INFO)

# ── Language detection via Unicode script ranges ───────────────────────────────
SCRIPT_RANGES = {
    "bn-IN": (0x0980, 0x09FF),
    "hi-IN": (0x0900, 0x097F),
    "te-IN": (0x0C00, 0x0C7F),
    "ta-IN": (0x0B80, 0x0BFF),
    "kn-IN": (0x0C80, 0x0CFF),
    "ml-IN": (0x0D00, 0x0D7F),
    "gu-IN": (0x0A80, 0x0AFF),
    "pa-IN": (0x0A00, 0x0A7F),
    "or-IN": (0x0B00, 0x0B7F),
}

def detect_language(text: str) -> str:
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


# ── Qdrant RAG retriever ───────────────────────────────────────────────────────
class RAGRetriever:
    def __init__(self):
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.oai             = openai.LLM.__new__(openai.LLM)
        self._openai_client  = None
        self.collection      = "abc_games"
        self.embed_model     = "text-embedding-3-small"

    async def _get_client(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai_client

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        try:
            client = await self._get_client()
            resp   = await client.embeddings.create(
                input=query, model=self.embed_model
            )
            vector  = resp.data[0].embedding
            results = self.qdrant.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=0.5,   # ignore low-relevance results
            )
            if not results:
                return ""
            context = "\n".join(
                f"- {r.payload['text']}" for r in results
            )
            logger.info(f"📚 RAG retrieved {len(results)} chunks for: {query[:50]}")
            return context
        except Exception as e:
            logger.warning(f"⚠️ RAG retrieval failed: {e}")
            return ""


INSTRUCTIONS = """
You are Mia from ABC Games — a friendly, persuasive voice agent calling to reactivate users.

## CRITICAL: USE ONLY PROVIDED CONTEXT
- You will receive VERIFIED FACTS from our database before each response
- NEVER make up offers, amounts, rules, or game details
- ONLY quote figures and rules that appear in the provided context
- If something is not in the context, say you'll check and get back to them

## LANGUAGE RULES (STRICT):
- ALWAYS reply in the EXACT same language the customer used
- Gujarati → Gujarati only | Hindi → Hindi only | Bengali → Bengali only
- English → English only | NEVER mix languages | NEVER switch on your own

## PITCH ORDER (follow across conversation):
1. Warm greeting + introduce as Mia from ABC Games
2. Note they haven't visited recently
3. Share the offer (use ONLY context-verified details)
4. Mention free spins
5. Create urgency (use context for exact duration)
6. Handle objections using context
7. Close politely

## STYLE:
- Max 1-2 short sentences per reply
- Warm and natural — never robotic
- Persuasive but not pushy
"""


class AIVoiceAgent(Agent):
    def __init__(self, rag: RAGRetriever) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self._rag          = rag
        self._current_lang = "en-IN"

    async def on_enter(self) -> None:
        logger.info("✅ on_enter — generating greeting")
        await self.session.generate_reply(
            instructions="Give a short warm single-sentence greeting. Introduce yourself as Mia from ABC Games. English only."
        )

    # ── RAG: inject context before LLM sees the message ──────────────────────
    async def llm_node(self, chat_ctx: ChatContext, tools, model_settings: ModelSettings):
        # Get last user message to query RAG
        user_msgs = [m for m in chat_ctx.messages if m.role == "user"]
        if user_msgs:
            last_user = str(user_msgs[-1].content)
            context   = await self._rag.retrieve(last_user)

            if context:
                # Prepend verified context as a system message
                rag_msg = ChatMessage.create(
                    role="system",
                    text=(
                        "VERIFIED FACTS FROM DATABASE — use only these for offer details:\n"
                        f"{context}\n\n"
                        "Do NOT invent any numbers, rules, or details not listed above."
                    ),
                )
                chat_ctx = chat_ctx.copy()
                chat_ctx.messages.insert(0, rag_msg)
                logger.info("📚 RAG context injected into LLM prompt")

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    # ── TTS: dynamic language switching ──────────────────────────────────────
    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
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

    for key in ["SARVAM_API_KEY", "OPENAI_API_KEY", "QDRANT_URL"]:
        if not os.getenv(key):
            logger.warning(f"⚠️ {key} not set")

    logger.info("✅ All keys checked")
    await ctx.connect()

    rag = RAGRetriever()

    session = AgentSession(
        # ✅ Silero VAD — filters noise/silence before STT
        # Cuts background noise, only sends real speech to Sarvam
        vad=silero.VAD.load(
            min_speech_duration=0.1,       # ignore very short sounds
            min_silence_duration=0.5,      # wait 500ms of silence before ending
            activation_threshold=0.6,      # higher = less sensitive to noise
            prefix_padding_duration=0.3,
        ),
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

    await session.start(agent=AIVoiceAgent(rag=rag), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
