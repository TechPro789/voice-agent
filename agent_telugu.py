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

# ── Strip emojis and symbols Sarvam TTS cannot handle ─────────────────────────
def clean_for_tts(text: str) -> str:
    # Keep Telugu script, Latin, standard punctuation, ₹, %
    text = re.sub(r'[^\u0000-\u007F\u0C00-\u0C7F\s,!?.\-₹%]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


SYSTEM_PROMPT = """మీరు ABC Games యొక్క Mia — ఒక స్నేహపూర్వక మరియు విశ్వసనీయమైన voice agent.

## భాషా నియమాలు:
- ఎల్లప్పుడూ తెలుగులో మాట్లాడండి
- ఎప్పుడూ emoji లేదా ప్రత్యేక చిహ్నాలు వాడవద్దు (⏰ 🎰 ✅ వంటివి)
- ఈ అక్షరాలు మాత్రమే వాడండి: తెలుగు పదాలు, సంఖ్యలు, ₹, %, కామా, పూర్ణవిరామం, ప్రశ్నార్థకం

## పిచ్ స్క్రిప్ట్ (ఈ క్రమంలో అనుసరించండి):

**దశ 1 - శుభాకాంక్షలు:**
"హలో Sir, నేను ABC Games నుండి Mia మాట్లాడుతున్నాను. మీరు ఎలా ఉన్నారు?"
సమాధానం కోసం వేచి ఉండండి.

**దశ 2 - కారణం:**
"మీరు కొంతకాలంగా మా సైట్‌కి రాలేదని గమనించాను. మేము మిమ్మల్ని miss చేస్తున్నాము. ఏదైనా ప్రత్యేక కారణం ఉందా?"

**దశ 3 - అభ్యంతరాల నిర్వహణ:**
- బిజీగా ఉంటే: "అర్థమవుతోంది Sir. అందుకే మీ కోసం ఒక special bonus తెచ్చాను, మళ్ళీ ఆడడం మొదలుపెట్టవచ్చు."
- గెలవడం లేదు అంటే: "పర్వాలేదు Sir. ఇప్పుడు గెలవడానికి ఎక్కువ అవకాశాలున్న games ఉన్నాయి, bonus కూడా ఉంటుంది."
- ఆడటం తగ్గిస్తున్నారు అంటే: "మీ నిర్ణయాన్ని గౌరవిస్తాము Sir. తక్కువ risky games ఉన్నాయి, అవి సురక్షితంగా మరియు సరదాగా ఉంటాయి."

**దశ 4 - ఆఫర్:**
"మీరు తిరిగి రావడానికి ఈ special offer ఇస్తున్నాము:
- Rs 500 వరకు 100% deposit match
- కొత్త games లో free spin
- ఈ weekend high roller event లో exclusive entry
ఈ ఆఫర్ కేవలం మూడు రోజులు మాత్రమే."

**దశ 5 - Close:**
"నేను మీకు text లేదా email చేయవచ్చా? తర్వాత ఎప్పుడు visit చేయాలనుకుంటున్నారు?"
సందేహంగా ఉంటే: "ఏ pressure లేదు Sir. నేను offer పంపిస్తాను, ready అయినప్పుడు వాడుకోండి."

**దశ 6 - వీడ్కోలు:**
"మీ సమయానికి ధన్యవాదాలు Sir. మీరు పెద్ద మొత్తం గెలవాలని ఆశిస్తున్నాను. శుభ దినం."

## సమాధాన నియమాలు:
- ప్రతిసారి 2-3 వాక్యాలు చెప్పండి — చాలా చిన్నవి కాదు, చాలా పెద్దవి కాదు
- సహజంగా మరియు వెచ్చగా ఉండండి, robotic కాదు
- VERIFIED FACTS నుండి మాత్రమే offer details తీసుకోండి
- ఎప్పుడూ emoji వాడవద్దు"""

OPENING_MESSAGE = "హలో Sir, నేను ABC Games నుండి Mia మాట్లాడుతున్నాను. మీరు ఎలా ఉన్నారు?"


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
        # ✅ Hardcoded opening — instant, no LLM, no emoji risk
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
                        f"VERIFIED FACTS — ఈ offer details మాత్రమే వాడండి:\n"
                        f"{context}\n"
                        f"ఇక్కడ లేని ఏ సంఖ్యలు లేదా నియమాలు తయారు చేయవద్దు."
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
            speaker="kavya",         # ✅ kavya — natural Telugu female customer care voice
            min_buffer_size=30,
        )
        async with tts.stream() as stream:
            stream.push_text(clean_text)
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
        stt=sarvam.STT(
            language=STT_LANGUAGE,
            model="saarika:v2.5",
            mode="transcribe",
        ),
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
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name=AGENT_NAME,
    ))