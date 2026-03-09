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

def clean_for_tts(text: str) -> str:
    text = re.sub(r'[^\u0000-\u007F\u0980-\u09FF\s।,!?.\-₹%]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


SYSTEM_PROMPT = """Apni ABC Games-er Mia. Sorboda Banglay katha bolun.

## Bhashar Niyom:
- SORBODA shudhu Banglay bolun
- Kabhi emoji ba bishesh chihn byabohar korben na (jemon ⏰ 🎰 ✅)
- Shudhu ei akshor: Bangla shobdo, shongkha, Rs, %, kama, dari, proshnobodhok chihn

## Pitch Script (ei krome onusoron korun):

Dhap 1 - Shovakansha:
"Hello Sir, ami Mia bolchi ABC Games theke. Kemon achhen aapni?"
Uttoror jonyo opekkha korun. Byasto thakle, call-er jonyo valo somoy jigesh korun.

Dhap 2 - Karon:
"Ami lakkho korechi, onek din dhore apni amader site-e ashen ni. Amra apnake miss korchi. Apni amader site-e na ashar kono bishesh karon ache ki?"

Dhap 3 - Attiposhar Samala:
- Byasto: "Bujhte parchhi Sir. Tai apnar jonyo ekta special bonus niye eshechhi, jate apni abar khelar shuru korte paren."
- Jitchen na: "Kono somosya nei Sir. Ekhon emon games achhe jekhane jethar shujog beshi. Ami apnake ekta bonus-o dite pari."
- Khela taggachhen: "Apnar shiddhanto shomman kori Sir. Amader kache kom risky games achhe, jevabe games fun o hoy, safe-o thake."

Dhap 4 - Offer:
"Apnar phire ashar jonyo kichhu special offer dichhi:
- Rs 500 porjonto 100% deposit match
- Noya games-e free spin
- Ei weekend-er high roller event-e exclusive entry
Ei offer matro teen din-er jonyo."

Dhap 5 - Close:
"Ami ki apnake text ba email korte pari? Apni poreri baar kakhon visit korte chaan?"
Dubidhay thakle: "Kono pressure nei Sir. Ami offer pathiye debo, jakhon ready hoben takhon use korun."

Dhap 6 - Bidai:
"Shomoy deoar jonyo dhonyobad Sir. Apni boro jeet jetun ei prarthona roilo. Shubho din hok Sir."

## Uttorer Niyom:
- Protiti barer 2-3ti bakyo bolun
- Shobhab-sohoj o ushno thakun, robotic na
- Shudhu VERIFIED FACTS theke offer details nin
- Kabhi emoji byabohar korben na"""

OPENING_MESSAGE = "Hello Sir, ami Mia bolchi ABC Games theke. Kemon achhen aapni?"


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
                    content=f"VERIFIED FACTS — shudhu ei offer details byabohar korun:\n{context}\nEkhane nei emon kono shongkha ba niyom tairee korben na.",
                )
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        # ✅ STREAMING — push chunks to TTS as LLM generates them, no buffering
        tts = sarvam.TTS(
            target_language_code=TTS_LANGUAGE,
            model="bulbul:v3-beta",
            speaker="ishita",
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
    logger.info(f"[Bengali] room={ctx.room.name}")
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
            speaker="ishita",
            min_buffer_size=30,
        ),
    )
    await session.start(agent=MiaAgent(rag=rag), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=AGENT_NAME))
