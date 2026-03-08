"""
Bengali Voice Agent — run: python agent_bengali.py start
"""
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
logger = logging.getLogger("agent-bengali")
logger.setLevel(logging.INFO)

STT_LANGUAGE  = "bn-IN"
TTS_LANGUAGE  = "bn-IN"
AGENT_NAME    = "bengali-agent"
SYSTEM_PROMPT = """আপনি ABC Games-এর Mia। সবসময় শুধুমাত্র বাংলায় কথা বলুন।
শুধুমাত্র দেওয়া তথ্য ব্যবহার করুন। পিচের ক্রম:
১. উষ্ণ অভিবাদন ও পরিচয়
২. অফার: ১০০% ডিপোজিট ম্যাচ, সর্বোচ্চ ৫০০ টাকা + ফ্রি স্পিন
৩. জরুরিতা: মাত্র ৩ দিন বৈধ
৪. আপত্তি সামলান ও বিনয়ের সাথে শেষ করুন
প্রতি উত্তরে সর্বোচ্চ ১-২ বাক্য। স্বাভাবিক ও উষ্ণ।"""
GREETING      = """বাংলায় একটি সংক্ষিপ্ত উষ্ণ অভিবাদন দিন এবং ABC Games-এর Mia হিসেবে পরিচয় দিন।"""


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
            resp    = await client.embeddings.create(input=query, model="text-embedding-3-small")
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
        await self.session.generate_reply(instructions=GREETING)

    async def llm_node(self, chat_ctx: ChatContext, tools, model_settings: ModelSettings):
        user_msgs = [m for m in chat_ctx.messages if m.role == "user"]
        if user_msgs:
            context = await self._rag.retrieve(str(user_msgs[-1].content))
            if context:
                rag_msg = ChatMessage.create(
                    role="system",
                    text=f"VERIFIED FACTS — use only these:\n{context}\nDo NOT invent numbers or rules.",
                )
                chat_ctx = chat_ctx.copy()
                chat_ctx.messages.insert(0, rag_msg)
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        chunks = []
        async for chunk in text:
            chunks.append(chunk)
        full_text = "".join(chunks).strip()
        if not full_text:
            return
        tts = sarvam.TTS(
            target_language_code=TTS_LANGUAGE,
            model="bulbul:v2",
            speaker="anushka",
            min_buffer_size=30,
        )
        async with tts.stream() as stream:
            stream.push_text(full_text)
            stream.end_input()
            async for ev in stream:
                yield ev.frame


# ✅ TOP-LEVEL function — required for multiprocessing pickle on Linux & Windows
async def entrypoint(ctx: JobContext):
    logger.info(f"[Bengali] room={ctx.room.name}")
    await ctx.connect()
    rag = RAGRetriever()
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.5,
            activation_threshold=0.6,
            prefix_padding_duration=0.3,
        ),
        stt=sarvam.STT(language=STT_LANGUAGE, model="saarika:v2.5", mode="transcribe"),
        llm=openai.LLM(model="gpt-4o", temperature=0.7),
        tts=sarvam.TTS(
            target_language_code=TTS_LANGUAGE,
            model="bulbul:v2",
            speaker="anushka",
            min_buffer_size=30,
        ),
    )
    await session.start(agent=MiaAgent(rag=rag), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=AGENT_NAME,
        # ✅ NO worker_type — causes AttributeError on livekit-agents 1.4.4
    ))
