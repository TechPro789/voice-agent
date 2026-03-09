"""
Telugu Voice Agent — run: python agent_telugu.py start
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
logger = logging.getLogger("agent-telugu")
logger.setLevel(logging.INFO)

STT_LANGUAGE = "te-IN"
TTS_LANGUAGE = "te-IN"
AGENT_NAME   = "telugu-agent"

def clean_for_tts(text: str) -> str:
    text = re.sub(r'[^\u0000-\u007F\u0C00-\u0C7F\s,!?.\-₹%]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


SYSTEM_PROMPT = """Meeru ABC Games yokka Mia. Eppudu Telugu lo matladandi.

## Language Rules:
- ALWAYS Telugu only
- NEVER use emoji or symbols like ⏰ 🎰 ✅
- Only use: Telugu words, numbers, Rs, %, comma, period, question mark

## Pitch Script (follow this order):

Step 1 - Opening:
"Hello Sir, nenu Mia matladutunnanu ABC Games nundi. Ela unnaru meeru?"
Wait for response. If busy, ask for a better time.

Step 2 - Reason:
"Nenu gamaninchanu, konni rojulu nundi meeru mana site ki visit cheyaledu. Memu mimmalni miss ayyamu. Meeru mana site ki visit cheyakapovadaniki emaina pratyeka reason unda ani check chesukuntunnanu."

Step 3 - Objections:
- Busy: "Nenu artham chesukuntunnanu Sir. Kani isari memu meeku oka special bonus istunnamu, meeru malli adadam start cheyavachu."
- Not winning: "No problem Sir. Ippudu chala games unnay, vaatilo gelavadaniki chances ekkuva. Nenu meeku oka bonus istanu."
- Playing less: "Memu me decision ki respect istunnamu Sir. Mana degara low-risk games unnay, avi thakkuva risky ga mariyu deposit limits limited. Games fun kuda untay, safe kuda untay."

Step 4 - Offer:
"Meeru malli join avvadaniki memu konni special offers istunnamu:
- Rs 500 varaku 100% deposit match
- Mana kotta games lo free spins
- Ee weekend high roller event lo exclusive entry
Ee offer kevalam moodu rojulu matrame untundi."

Step 5 - Close:
"Nenu meeku text leka email pampagalanaa? Next time visit cheyadaniki eppudu convenient?"
If hesitant: "No pressure Sir. Nenu meeku offer pampistanu, meeru ready ainappudu use cheyandi."

Step 6 - Ending:
"Mee time icchinanduku thanks Sir. Meeru pedda gelupu gelavalani korukuntunnamu. Have a good day Sir."

## Response Rules:
- 2-3 sentences per turn
- Natural and warm, not robotic
- Use only VERIFIED FACTS for offer details
- NEVER use emoji"""

OPENING_MESSAGE = "Hello Sir, nenu Mia matladutunnanu ABC Games nundi. Ela unnaru meeru?"


class RAGRetriever:
    def __init__(self):
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self._oai_client = None

    async def _get_client(self):
        if self._oai_client is None:
            from openai import AsyncOpenAI
            self._oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._oai_client

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
        await self.session.say(OPENING_MESSAGE)

    async def llm_node(self, chat_ctx: ChatContext, tools, model_settings: ModelSettings):
        all_msgs  = chat_ctx.messages()
        user_msgs = [m for m in all_msgs if m.role == "user"]
        if user_msgs:
            context = await self._rag.retrieve(str(user_msgs[-1].content))
            if context:
                chat_ctx = chat_ctx.copy()
                chat_ctx.add_message(
                    role="system",
                    content=f"VERIFIED FACTS — use only these:\n{context}\nDo NOT invent numbers or rules.",
                )
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        # ✅ STREAMING — push chunks to TTS as LLM generates them, no buffering
        tts = sarvam.TTS(
            target_language_code=TTS_LANGUAGE,
            model="bulbul:v3-beta",
            speaker="kavya",
            min_buffer_size=30,
        )
        async with tts.stream() as stream:
            async for chunk in text:
                clean = clean_for_tts(chunk)
                if clean:
                    stream.push_text(clean)
            stream.end_input()
            async for ev in stream:
                yield ev.frame


async def entrypoint(ctx: JobContext):
    logger.info(f"[Telugu] room={ctx.room.name}")
    await ctx.connect()
    rag = RAGRetriever()
    session = AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.05,
            min_silence_duration=0.3,
            activation_threshold=0.55,
            prefix_padding_duration=0.2,
        ),
        stt=sarvam.STT(language=STT_LANGUAGE, model="saarika:v2.5", mode="transcribe"),
        llm=openai.LLM(model="gpt-4o", temperature=0.7),
        tts=sarvam.TTS(
            target_language_code=TTS_LANGUAGE,
            model="bulbul:v3-beta",
            speaker="kavya",
            min_buffer_size=30,
        ),
    )
    await session.start(agent=MiaAgent(rag=rag), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=AGENT_NAME))
