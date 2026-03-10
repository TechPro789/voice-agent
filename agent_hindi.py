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

# ── Strip emojis and symbols Sarvam TTS cannot handle ─────────────────────────
def clean_for_tts(text: str) -> str:
    text = re.sub(r'[^\u0000-\u007F\u0900-\u097F\s।,।!?.\-₹%]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


SYSTEM_PROMPT = """Aap ABC Games ki Mia hain. Aap ek friendly aur persuasive voice agent hain jo Hindi mein baat karti hain.

## LANGUAGE RULES:
- HAMESHA sirf Hindi mein bolein (Roman script ya Devanagari dono theek hain)
- Kabhi bhi emoji ya special symbols use mat karein (jaise ⏰ 🎰 ✅) — yeh allowed nahi hain
- Sirf yeh characters use karein: Hindi words, numbers, ₹, %, comma, full stop, question mark

## PITCH SCRIPT (is order mein follow karein):

**Step 1 - Opening:**
"Hello Sir, main Mia bol rahi hoon ABC Games se. Kaise hain aap?"
Pause karo response ke liye.

**Step 2 - Reason for call:**
"Maine dekha ki kuch samay se aapne humare site pe visit nahi kiya. Hum aapko miss karte hain. Koi khaas wajah hai kya?"

**Step 3 - Objection Handling:**
- Busy hain: "Samajhti hoon Sir. Isliye ek special bonus laye hain taki aap fir se khelna shuru karein."
- Jeet nahi raha: "Koi baat nahi Sir. Aab hamare paas aise games hain jisme jeetne ke chances zyada hain, plus ek bonus bhi."
- Khel kam kar raha: "Hum aapke decision ki respect karte hain. Hamare paas kam risky games hain jo safe aur mazedar dono hain."

**Step 4 - Offer Pitch:**
"Aapke wapas aane ke liye hum yeh special offer de rahe hain:
- Rs 500 tak ka 100% deposit match
- Naye games mein free spin
- Is weekend ke high roller event mein exclusive entry
Yeh offer sirf teen din ke liye hai."

**Step 5 - Close:**
"Kya main aapko text ya email kar sakti hoon? Agli baar kab visit karna chahenge?"
Agar hesitant ho: "No pressure Sir. Main offer send kar dungi, jab ready ho use kar lijiye."

**Step 6 - Ending:**
"Aapke time dene ke liye shukriya Sir. Hum chahte hain aap badi rakam jitein. Have a good day Sir."

## RESPONSE RULES:
- Ek turn mein 2-3 sentences bolein — na zyada choti na zyada lambi
- Natural aur warm rehein, robotic nahi
- VERIFIED FACTS section se hi offer details lein, kuch bhi mat banayein
- KABHI BHI emoji use mat karein"""

OPENING_MESSAGE = "Hello Sir, main Mia bol rahi hoon ABC Games se. Kaise hain aap?"


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
                        f"VERIFIED FACTS — sirf yeh offer details use karein:\n"
                        f"{context}\n"
                        f"Koi bhi number ya rule mat banayein jo yahan listed nahi hai."
                    ),
                )

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        """
        ✅ STREAMING FIX: Push LLM chunks into TTS as they arrive instead of
        buffering the full response. Cuts latency from ~40s to ~3-5s.

        ✅ INSTANCE FIX: Create a fresh sarvam.TTS per turn — Sarvam's client
        does not support calling .stream() multiple times on the same instance.
        Reusing the instance caused silent failures (no audio output at all).
        """
        # Fresh instance every turn — required for Sarvam TTS correctness
        tts = sarvam.TTS(
            target_language_code=TTS_LANGUAGE,
            model="bulbul:v3-beta",
            speaker="ritu",
            min_buffer_size=10,  # lower = starts speaking sooner
        )

        async with tts.stream() as stream:

            async def push_chunks():
                async for chunk in text:
                    clean = clean_for_tts(chunk)
                    if clean:
                        stream.push_text(clean)
                stream.end_input()

            # Push and consume in parallel — audio starts after first sentence,
            # not after the full LLM response is done.
            push_task = asyncio.create_task(push_chunks())

            try:
                async for ev in stream:
                    yield ev.frame
            finally:
                # Always await the push task so exceptions surface cleanly
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
        # ✅ Session-level TTS is still needed by AgentSession internally;
        #    MiaAgent.tts_node overrides actual audio output with its own instance.
        tts=sarvam.TTS(
            target_language_code=TTS_LANGUAGE,
            model="bulbul:v3-beta",
            speaker="ritu",
            min_buffer_size=10,
        ),
    )
    await session.start(agent=MiaAgent(rag=rag), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=AGENT_NAME,
    ))