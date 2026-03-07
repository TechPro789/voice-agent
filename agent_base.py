"""
Base agent — shared logic for all language instances.
Import this in each language-specific agent file.
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
logger = logging.getLogger("ai-voice-agent")
logger.setLevel(logging.INFO)


# ── RAG Retriever ──────────────────────────────────────────────────────────────
class RAGRetriever:
    def __init__(self):
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self._client    = None
        self.collection = "abc_games"

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    async def retrieve(self, query: str, top_k: int = 3) -> str:
        try:
            client   = await self._get_client()
            resp     = await client.embeddings.create(input=query, model="text-embedding-3-small")
            vector   = resp.data[0].embedding
            results  = self.qdrant.search(
                collection_name=self.collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=0.5,
            )
            if not results:
                return ""
            context = "\n".join(f"- {r.payload['text']}" for r in results)
            logger.info(f"📚 RAG: {len(results)} chunks retrieved")
            return context
        except Exception as e:
            logger.warning(f"⚠️ RAG failed: {e}")
            return ""


# ── Language Config ────────────────────────────────────────────────────────────
LANGUAGE_CONFIGS = {
    "bengali": {
        "stt_language":   "bn-IN",
        "tts_language":   "bn-IN",
        "tts_speaker":    "anushka",
        "agent_name":     "bengali-agent",
        "room_name":      "bengali-room",
        "system_prompt":  """
আপনি ABC Games-এর Mia। আপনি বাংলায় ব্যবহারকারীদের সাথে কথা বলছেন।

## নিয়ম:
- সবসময় শুধুমাত্র বাংলায় কথা বলুন
- ইংরেজি বা অন্য কোনো ভাষা ব্যবহার করবেন না
- শুধুমাত্র দেওয়া তথ্য ব্যবহার করুন, কিছু বানাবেন না

## পিচের ক্রম:
১. উষ্ণ অভিবাদন ও পরিচয়
২. সম্প্রতি না আসার উল্লেখ
৩. অফার: ১০০% ডিপোজিট ম্যাচ, সর্বোচ্চ ৫০০ টাকা
৪. ফ্রি স্পিনের উল্লেখ
৫. জরুরিতা: মাত্র ৩ দিন বৈধ
৬. ভদ্রভাবে সমাপ্ত করুন

## স্টাইল:
- প্রতি উত্তরে সর্বোচ্চ ১-২ বাক্য
- স্বাভাবিক ও উষ্ণ — রোবটিক নয়
- প্ররোচনামূলক কিন্তু চাপাচাপি নয়
""",
        "greeting": "বাংলায় একটি সংক্ষিপ্ত ও উষ্ণ অভিবাদন দিন এবং ABC Games-এর Mia হিসেবে পরিচয় দিন।",
    },

    "hindi": {
        "stt_language":   "hi-IN",
        "tts_language":   "hi-IN",
        "tts_speaker":    "anushka",
        "agent_name":     "hindi-agent",
        "room_name":      "hindi-room",
        "system_prompt":  """
आप ABC Games की Mia हैं। आप हिंदी में उपयोगकर्ताओं से बात कर रही हैं।

## नियम:
- हमेशा केवल हिंदी में बोलें
- अंग्रेजी या कोई अन्य भाषा का उपयोग न करें
- केवल दी गई जानकारी का उपयोग करें, कुछ भी न बनाएं

## पिच का क्रम:
१. गर्मजोशी से अभिवादन और परिचय
२. हाल ही में न आने का उल्लेख
३. ऑफर: १००% डिपॉजिट मैच, अधिकतम ₹५००
४. फ्री स्पिन का उल्लेख
५. अर्जेंसी: केवल ३ दिन वैध
६. विनम्रता से समाप्त करें

## शैली:
- प्रति उत्तर अधिकतम १-२ वाक्य
- स्वाभाविक और गर्म — रोबोटिक नहीं
- प्रेरक लेकिन दबाव नहीं
""",
        "greeting": "हिंदी में एक संक्षिप्त और गर्मजोशी भरा अभिवादन दें और ABC Games की Mia के रूप में परिचय दें।",
    },

    "telugu": {
        "stt_language":   "te-IN",
        "tts_language":   "te-IN",
        "tts_speaker":    "anushka",
        "agent_name":     "telugu-agent",
        "room_name":      "telugu-room",
        "system_prompt":  """
మీరు ABC Games యొక్క Mia. మీరు తెలుగులో వినియోగదారులతో మాట్లాడుతున్నారు.

## నియమాలు:
- ఎల్లప్పుడూ తెలుగులో మాత్రమే మాట్లాడండి
- ఆంగ్లం లేదా ఇతర భాషలు ఉపయోగించవద్దు
- అందించిన సమాచారం మాత్రమే ఉపయోగించండి

## పిచ్ క్రమం:
౧. వెచ్చని శుభాకాంక్షలు మరియు పరిచయం
౨. ఇటీవల రాలేదని పేర్కొనండి
౩. ఆఫర్: ౧౦౦% డిపాజిట్ మ్యాచ్, గరిష్టంగా ₹౫౦౦
౪. ఫ్రీ స్పిన్స్ పేర్కొనండి
౫. అర్జెన్సీ: కేవలం ౩ రోజులు మాత్రమే
౬. మర్యాదగా ముగించండి

## స్టైల్:
- ప్రతి సమాధానంలో గరిష్టంగా ౧-౨ వాక్యాలు
- సహజంగా మరియు వెచ్చగా — రోబోటిక్ కాదు
""",
        "greeting": "తెలుగులో ఒక సంక్షిప్త మరియు వెచ్చని శుభాకాంక్ష చెప్పండి మరియు ABC Games యొక్క Mia గా పరిచయం చేసుకోండి.",
    },
}


# ── Base Agent Class ───────────────────────────────────────────────────────────
class MiaAgent(Agent):
    def __init__(self, config: dict, rag: RAGRetriever) -> None:
        super().__init__(instructions=config["system_prompt"])
        self._config = config
        self._rag    = rag

    async def on_enter(self) -> None:
        logger.info(f"✅ [{self._config['agent_name']}] on_enter")
        await self.session.generate_reply(
            instructions=self._config["greeting"]
        )

    async def llm_node(self, chat_ctx: ChatContext, tools, model_settings: ModelSettings):
        user_msgs = [m for m in chat_ctx.messages if m.role == "user"]
        if user_msgs:
            last_user = str(user_msgs[-1].content)
            context   = await self._rag.retrieve(last_user)
            if context:
                rag_msg = ChatMessage.create(
                    role="system",
                    text=(
                        "VERIFIED FACTS — use only these for offer details:\n"
                        f"{context}\n\nDo NOT invent any numbers or rules."
                    ),
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

        tts_instance = sarvam.TTS(
            target_language_code=self._config["tts_language"],
            model="bulbul:v2",
            speaker=self._config["tts_speaker"],
            min_buffer_size=30,
        )
        async with tts_instance.stream() as stream:
            stream.push_text(full_text)
            stream.end_input()
            async for ev in stream:
                yield ev.frame


# ── Shared entrypoint factory ──────────────────────────────────────────────────
def make_entrypoint(language: str):
    config = LANGUAGE_CONFIGS[language]

    async def entrypoint(ctx: JobContext):
        logger.info(f"[{language.upper()}] User joined: {ctx.room.name}")
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
                language=config["stt_language"],  # fixed language — no auto-detect
                model="saarika:v2.5",
                mode="transcribe",
            ),
            llm=openai.LLM(model="gpt-4o", temperature=0.7),
            tts=sarvam.TTS(
                target_language_code=config["tts_language"],
                model="bulbul:v2",
                speaker=config["tts_speaker"],
                min_buffer_size=30,
            ),
        )

        await session.start(
            agent=MiaAgent(config=config, rag=rag),
            room=ctx.room,
        )

    return entrypoint
