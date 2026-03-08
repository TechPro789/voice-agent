"""
Bengali Voice Agent — run: python agent_bengali.py start
"""
import os
import re
import logging
from typing import AsyncIterable
from dotenv import load_dotenv

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession, ModelSettings
from livekit.agents.llm import ChatContext
from livekit.plugins import openai, sarvam, silero
from qdrant_client import QdrantClient

load_dotenv()
logger = logging.getLogger("agent-bengali")
logger.setLevel(logging.INFO)

STT_LANGUAGE = "bn-IN"
TTS_LANGUAGE = "bn-IN"
AGENT_NAME   = "bengali-agent"

# ── Strip emojis and symbols Sarvam TTS cannot handle ─────────────────────────
def clean_for_tts(text: str) -> str:
    # Keep Bengali script, Latin, standard punctuation, ₹, %
    text = re.sub(r'[^\u0000-\u007F\u0980-\u09FF\s।,!?.\-₹%]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


SYSTEM_PROMPT = """আপনি ABC Games-এর Mia — একজন বন্ধুত্বপূর্ণ এবং বিশ্বাসযোগ্য voice agent।

## ভাষার নিয়ম:
- সবসময় শুধুমাত্র বাংলায় কথা বলুন
- কখনো emoji বা বিশেষ চিহ্ন ব্যবহার করবেন না (যেমন ⏰ 🎰 ✅)
- শুধুমাত্র এই অক্ষর ব্যবহার করুন: বাংলা শব্দ, সংখ্যা, ₹, %, কমা, দাড়ি, প্রশ্নবোধক চিহ্ন

## পিচ স্ক্রিপ্ট (এই ক্রমে অনুসরণ করুন):

**ধাপ ১ - শুভেচ্ছা:**
"হ্যালো Sir, আমি Mia বলছি ABC Games থেকে। কেমন আছেন আপনি?"
উত্তরের জন্য অপেক্ষা করুন।

**ধাপ ২ - কারণ:**
"আমি দেখলাম কিছুদিন ধরে আপনি আমাদের সাইটে আসেননি। আমরা আপনাকে miss করছি। কোনো বিশেষ কারণ আছে কি?"

**ধাপ ৩ - আপত্তি সামলানো:**
- ব্যস্ত থাকলে: "বুঝতে পারছি Sir। তাই আপনার জন্য একটি special bonus নিয়ে এসেছি যাতে আবার খেলা শুরু করতে পারেন।"
- জিততে পারছেন না: "কোনো সমস্যা নেই Sir। এখন এমন games আছে যেখানে জেতার সুযোগ বেশি, সাথে bonus-ও পাবেন।"
- খেলা কমাচ্ছেন: "আমরা আপনার সিদ্ধান্তকে সম্মান করি Sir। আমাদের কাছে কম risky games আছে যা নিরাপদ এবং মজাদার।"

**ধাপ ৪ - অফার:**
"আপনার ফিরে আসার জন্য আমরা special offer দিচ্ছি:
- Rs 500 পর্যন্ত 100% deposit match
- নতুন games-এ free spin
- এই weekend-এর high roller event-এ exclusive entry
এই অফার মাত্র তিন দিনের জন্য।"

**ধাপ ৫ - Close:**
"আমি কি আপনাকে text বা email করতে পারি? পরের বার কখন visit করতে চান?"
দ্বিধা থাকলে: "কোনো pressure নেই Sir। আমি offer পাঠিয়ে দেব, যখন ready হবেন তখন ব্যবহার করবেন।"

**ধাপ ৬ - বিদায়:**
"সময় দেওয়ার জন্য ধন্যবাদ Sir। আশা করি আপনি বড় পরিমাণ জিতবেন। শুভ দিন হোক।"

## প্রতিক্রিয়ার নিয়ম:
- প্রতি বার ২-৩টি বাক্য বলুন — খুব ছোট না, খুব বড় না
- স্বাভাবিক ও উষ্ণ থাকুন, robotic নয়
- VERIFIED FACTS থেকেই offer details নিন, কিছু বানাবেন না
- কখনো emoji ব্যবহার করবেন না"""

OPENING_MESSAGE = "হ্যালো Sir, আমি Mia বলছি ABC Games থেকে। কেমন আছেন আপনি?"


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
        # ✅ Hardcoded opening — no LLM, no emoji risk, instant
        await self.session.say(OPENING_MESSAGE)

    async def llm_node(self, chat_ctx: ChatContext, tools, model_settings: ModelSettings):
        # ✅ messages() — method call with parentheses
        all_msgs  = chat_ctx.messages()
        user_msgs = [m for m in all_msgs if m.role == "user"]

        if user_msgs:
            last_text = str(user_msgs[-1].content)
            context   = await self._rag.retrieve(last_text)
            if context:
                # ✅ add_message — correct API
                chat_ctx = chat_ctx.copy()
                chat_ctx.add_message(
                    role="system",
                    content=(
                        f"VERIFIED FACTS — শুধুমাত্র এই offer details ব্যবহার করুন:\n"
                        f"{context}\n"
                        f"এখানে listed নেই এমন কোনো number বা rule বানাবেন না।"
                    ),
                )

        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        chunks = []
        async for chunk in text:
            chunks.append(chunk)
        full_text = "".join(chunks).strip()
        if not full_text:
            return

        # ✅ Strip emojis before sending to Sarvam
        clean_text = clean_for_tts(full_text)
        if not clean_text:
            logger.warning(f"Text empty after cleaning: {full_text[:50]}")
            return

        logger.info(f"TTS [{TTS_LANGUAGE}]: {clean_text[:80]}")

        tts = sarvam.TTS(
            target_language_code=TTS_LANGUAGE,
            model="bulbul:v3-beta",  # ✅ v3-beta — better accent, customer care trained
            speaker="ishita",        # ✅ ishita — natural Bengali female voice
            min_buffer_size=30,
        )
        async with tts.stream() as stream:
            stream.push_text(clean_text)
            stream.end_input()
            async for ev in stream:
                yield ev.frame


async def entrypoint(ctx: JobContext):
    logger.info(f"[Bengali] room={ctx.room.name}")
    await ctx.connect()
    rag = RAGRetriever()
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.05,
            min_silence_duration=0.3,    # ✅ faster response
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
            speaker="ishita",
            min_buffer_size=30,
        ),
    )
    await session.start(agent=MiaAgent(rag=rag), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=AGENT_NAME,
    ))