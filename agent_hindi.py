"""
Hindi Voice Agent — run: python agent_hindi.py start
"""
import os
import re
import asyncio
import logging
from typing import AsyncIterable
from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession, ModelSettings
from livekit.agents.llm import ChatContext
from livekit.plugins import openai, sarvam, silero
from qdrant_client import QdrantClient

load_dotenv()
logger = logging.getLogger("agent-hindi")
logger.setLevel(logging.INFO)

STT_LANGUAGE = "hi-IN"
TTS_LANGUAGE = "hi-IN"
AGENT_NAME   = "hindi-agent"

# Sentence boundary punctuation for Hindi (।) and Latin (.!?)
SENTENCE_END = re.compile(r'[।.!?]+')

# ── Strip emojis and symbols Sarvam TTS cannot handle ─────────────────────────
def clean_for_tts(text: str) -> str:
    text = re.sub(r'[^\u0000-\u007F\u0900-\u097F\s।,!?.\-₹%]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


SYSTEM_PROMPT = """आप ABC Games की Mia हैं। आप एक friendly और persuasive voice agent हैं।

## भाषा के नियम (बहुत ज़रूरी):
- हमेशा सिर्फ देवनागरी लिपि में लिखें — जैसे "कैसे हैं आप?" न कि "Kaise hain aap?"
- कभी भी Roman script में Hindi मत लिखें
- कभी भी emoji या special symbols use मत करें (जैसे ⏰ 🎰 ✅)
- सिर्फ यह characters use करें: हिंदी शब्द, numbers, ₹, %, comma, full stop, question mark
- अंग्रेज़ी के technical शब्द जैसे "bonus", "deposit", "offer", "spin" को जस का तस लिखें — उनका अनुवाद मत करें

## PITCH SCRIPT (इसी क्रम में follow करें):

**Step 1 - शुरुआत:**
"नमस्ते सर, मैं Mia बोल रही हूँ ABC Games से। कैसे हैं आप?"
जवाब का इंतज़ार करें।

**Step 2 - Call का कारण:**
"मैंने देखा कि कुछ समय से आपने हमारी site पर visit नहीं किया। हम आपको miss करते हैं। कोई खास वजह है क्या?"

**Step 3 - Objection Handling:**
- Busy हैं: "समझती हूँ सर। इसीलिए एक special bonus लाई हूँ ताकि आप फिर से खेलना शुरू करें।"
- जीत नहीं रहा: "कोई बात नहीं सर। अब हमारे पास ऐसे games हैं जिनमें जीतने के chances ज़्यादा हैं, साथ में एक bonus भी।"
- खेल कम कर रहा: "हम आपके decision की respect करते हैं। हमारे पास कम risky games हैं जो safe और मज़ेदार दोनों हैं।"

**Step 4 - Offer:**
"आपके वापस आने के लिए हम यह special offer दे रहे हैं:
Rs 500 तक का 100% deposit match, नए games में free spin, और इस weekend के high roller event में exclusive entry।
यह offer सिर्फ तीन दिन के लिए है।"

**Step 5 - Close:**
"क्या मैं आपको text या email कर सकती हूँ? अगली बार कब visit करना चाहेंगे?"
अगर hesitant हों: "No pressure सर। मैं offer send कर दूँगी, जब ready हों तब use कर लीजिए।"

**Step 6 - अंत:**
"आपका समय देने के लिए शुक्रिया सर। हम चाहते हैं आप बड़ी रकम जीतें। Have a good day सर।"

## RESPONSE के नियम:
- एक turn में 2-3 sentences बोलें — न बहुत छोटा न बहुत लंबा
- Natural और warm रहें, robotic नहीं
- VERIFIED FACTS section से ही offer details लें, कुछ भी मत बनाएं
- कभी भी emoji use मत करें"""

OPENING_MESSAGE = "नमस्ते सर, मैं Mia बोल रही हूँ ABC Games से। कैसे हैं आप?"


class RAGRetriever:
    def __init__(self):
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self._client = None

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        try:
            client  = await self._get_client()
            resp    = await client.embeddings.create(
                input=query, model="text-embedding-3-small"
            )
            vector  = resp.data[0].embedding
            results = self.qdrant.search(
                collection_name="abc_games",
                query_vector=vector,
                limit=top_k,
                score_threshold=0.5,
            )
            if not results:
                return ""
            return "\n".join(f"- {r.payload['text']}" for r in results)
        except Exception as e:
            logger.warning(f"RAG failed: {e}")
            return ""


class MiaAgent(Agent):
    def __init__(self, rag: RAGRetriever) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)
        self._rag = rag

    async def on_enter(self) -> None:
        await self.session.say(OPENING_MESSAGE)

    async def llm_node(self, chat_ctx: ChatContext, tools, model_settings: ModelSettings):
        all_msgs  = chat_ctx.messages()
        user_msgs = [m for m in all_msgs if m.role == "user"]

        if user_msgs:
            last_text = str(user_msgs[-1].content)
            context   = await self._rag.retrieve(last_text)
            if context:
                chat_ctx = chat_ctx.copy()
                chat_ctx.add_message(
                    role="system",
                    content=(
                        f"VERIFIED FACTS — सिर्फ यह offer details use करें:\n"
                        f"{context}\n"
                        f"कोई भी number या rule मत बनाएं जो यहाँ listed नहीं है।"
                    ),
                )

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        """
        Sentence-boundary streaming — buffer tokens until a full sentence
        is ready before pushing to Sarvam. This gives the prosody model
        enough context for natural Hindi intonation.

        Speaker changed from 'ritu' to 'meera' — meera handles Devanagari
        input more naturally with consistent accent throughout.
        """
        # meera: better native Hindi prosody than ritu for Devanagari input
        tts = sarvam.TTS(
            target_language_code=TTS_LANGUAGE,
            model="bulbul:v3-beta",
            speaker="meera",
            min_buffer_size=30,
        )

        async with tts.stream() as stream:

            async def push_sentences():
                buffer = ""
                async for chunk in text:
                    clean = clean_for_tts(chunk)
                    if not clean:
                        continue
                    buffer += clean

                    while True:
                        match = SENTENCE_END.search(buffer)
                        if not match:
                            break
                        sentence = buffer[:match.end()].strip()
                        buffer   = buffer[match.end():].strip()
                        if sentence:
                            logger.info(f"TTS > {sentence[:80]}")
                            stream.push_text(sentence + " ")

                # Flush trailing text without punctuation
                if buffer.strip():
                    logger.info(f"TTS > (remainder) {buffer[:80]}")
                    stream.push_text(buffer.strip())

                stream.end_input()

            push_task = asyncio.create_task(push_sentences())

            try:
                async for ev in stream:
                    yield ev.frame
            finally:
                await push_task


async def entrypoint(ctx: JobContext):
    logger.info(f"[Hindi] room={ctx.room.name}")
    await ctx.connect()
    rag = RAGRetriever()
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.05,
            min_silence_duration=0.3,
            activation_threshold=0.55,
            prefix_padding_duration=0.2,
        ),
        stt=sarvam.STT(
            language=STT_LANGUAGE,
            model="saarika:v2.5",
            mode="transcribe",
        ),
        llm=openai.LLM(model="gpt-4o", temperature=0.7),
        tts=sarvam.TTS(
            target_language_code=TTS_LANGUAGE,
            model="bulbul:v3-beta",
            speaker="meera",
            min_buffer_size=30,
        ),
    )
    await session.start(agent=MiaAgent(rag=rag), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=AGENT_NAME,
    ))