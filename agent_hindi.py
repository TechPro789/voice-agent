"""
Hindi Voice Agent — run: python agent_hindi.py start
"""
import os
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

SYSTEM_PROMPT = """आप ABC Games की Mia हैं। हमेशा केवल हिंदी में बोलें।
केवल दी गई जानकारी का उपयोग करें। पिच का क्रम:
१. गर्मजोशी से अभिवादन और परिचय
२. ऑफर: १००% डिपॉजिट मैच, अधिकतम ₹५०० + फ्री स्पिन
३. अर्जेंसी: केवल ३ दिन वैध
४. आपत्तियां संभालें और विनम्रता से समाप्त करें
प्रति उत्तर अधिकतम १-२ वाक्य। स्वाभाविक और गर्म।"""

GREETING = "हिंदी में एक संक्षिप्त गर्मजोशी भरा अभिवादन दें और ABC Games की Mia के रूप में परिचय दें।"


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
        await self.session.generate_reply(instructions=GREETING)

    async def llm_node(self, chat_ctx: ChatContext, tools, model_settings: ModelSettings):
        # ✅ chat_ctx.messages() — method call with parentheses
        all_msgs  = chat_ctx.messages()
        user_msgs = [m for m in all_msgs if m.role == "user"]

        if user_msgs:
            last_text = str(user_msgs[-1].content)
            context   = await self._rag.retrieve(last_text)
            if context:
                # ✅ copy + add_message — correct API for livekit-agents 1.4.4
                chat_ctx = chat_ctx.copy()
                chat_ctx.add_message(
                    role="system",
                    content=(
                        f"VERIFIED FACTS — use only these for offer details:\n"
                        f"{context}\n"
                        f"Do NOT invent any numbers or rules not listed above."
                    ),
                )
                logger.info(f"📚 RAG injected {len(context)} chars")

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


async def entrypoint(ctx: JobContext):
    logger.info(f"[Hindi] room={ctx.room.name}")
    await ctx.connect()
    rag = RAGRetriever()
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.5,
            activation_threshold=0.6,
            prefix_padding_duration=0.3,
        ),
        stt=sarvam.STT(
            language=STT_LANGUAGE,
            model="saarika:v2.5",
            mode="transcribe",
        ),
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
    ))
