"""
Run ONCE locally: python ingest.py
Populates Qdrant with Hindi + Bengali + Telugu knowledge base.
"""
import os
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from openai import AsyncOpenAI

load_dotenv()

DOCUMENTS = [
    # ═══════════════════════════════════════════════════════════════════
    # HINDI (from Outbound Hindi Reactivate script)
    # ═══════════════════════════════════════════════════════════════════
    {"id": "hi_offer_deposit",     "language": "hindi",   "category": "offer",
     "text": "ABC Games Rs 500 tak ka 100% deposit match deta hai. Rs 500 deposit karo to Rs 500 extra milega — total Rs 1000 khelne ke liye."},
    {"id": "hi_offer_freespin",    "language": "hindi",   "category": "offer",
     "text": "Wapas aane wale users ko naye games mein free spin milte hain reactivation offer ke saath."},
    {"id": "hi_offer_highroller",  "language": "hindi",   "category": "offer",
     "text": "Is weekend ABC Games ka high roller event hai. Wapas aane wale users ko exclusive entry milegi."},
    {"id": "hi_offer_urgency",     "language": "hindi",   "category": "offer",
     "text": "Yeh offer sirf teen din ke liye valid hai. Teen din baad expire ho jaata hai."},
    {"id": "hi_offer_eligibility", "language": "hindi",   "category": "offer",
     "text": "Offer un users ke liye hai jo last 30 din mein visit nahi kiye. Bonus ke liye minimum Rs 100 deposit karna hoga."},
    {"id": "hi_obj_busy",          "language": "hindi",   "category": "objection",
     "text": "Agar user busy hai: Samajhti hoon Sir, life busy ho jaata hai. Isliye ek special bonus laye hain taki aap fir se khelna shuru karein."},
    {"id": "hi_obj_notwinning",    "language": "hindi",   "category": "objection",
     "text": "Agar user jeet nahi raha: Koi baat nahi Sir, ab aise games hain jisme jeetne ke chances zyada hain. Ek bonus bhi de sakti hoon."},
    {"id": "hi_obj_cuttingdown",   "language": "hindi",   "category": "objection",
     "text": "Agar user khel kam kar raha: Hum aapke decision ki respect karte hain Sir. Hamare paas low-risk games hain jo safe bhi hain aur mazedar bhi."},
    {"id": "hi_obj_notinterested", "language": "hindi",   "category": "objection",
     "text": "No pressure Sir. Main offer send kar dungi, jab ready ho tab use kar lijiye. Offer sirf teen din valid hai."},
    {"id": "hi_faq_claim",         "language": "hindi",   "category": "faq",
     "text": "Offer claim karne ke liye: Login karein, Promotions mein jayein, Claim Offer click karein, Rs 100 deposit karein. Bonus turant milta hai."},
    {"id": "hi_faq_trust",         "language": "hindi",   "category": "faq",
     "text": "ABC Games fully licensed aur secure hai. Transactions encrypted hain. Crores ki payouts ki hain. Support 24/7 available hai."},
    {"id": "hi_closing",           "language": "hindi",   "category": "closing",
     "text": "Kya main aapko text ya email kar sakti hoon? Agli baar kab visit karenge? No pressure Sir, offer send kar dungi jab ready ho tab use karein."},

    # ═══════════════════════════════════════════════════════════════════
    # BENGALI (from Outbound Bengali Reactivate script)
    # ═══════════════════════════════════════════════════════════════════
    {"id": "bn_offer_deposit",     "language": "bengali", "category": "offer",
     "text": "ABC Games Rs 500 porjonto 100% deposit match dey. Rs 500 deposit korle Rs 500 extra paben — mot Rs 1000 khelar jonyo."},
    {"id": "bn_offer_freespin",    "language": "bengali", "category": "offer",
     "text": "Noya games-e free spin milbe reactivation offer-er shathe phire aasha users-der."},
    {"id": "bn_offer_highroller",  "language": "bengali", "category": "offer",
     "text": "Ei weekend ABC Games-er high roller event achhe. Phire aasha users-der exclusive entry dewa hobe."},
    {"id": "bn_offer_urgency",     "language": "bengali", "category": "offer",
     "text": "Ei offer matro teen din-er jonyo valid. Teen din por automatically expire hoye jabe."},
    {"id": "bn_offer_eligibility", "language": "bengali", "category": "offer",
     "text": "Offer tar jonyo jo users sesh 30 din-e visit koren ni. Bonus-er jonyo minimum Rs 100 deposit korte hobe."},
    {"id": "bn_obj_busy",          "language": "bengali", "category": "objection",
     "text": "Jodi user byasto: Bujhte parchhi Sir, life byasto hoye jay. Tai apnar jonyo ekta special bonus niye eshechhi jate abar khelar shuru korte paren."},
    {"id": "bn_obj_notwinning",    "language": "bengali", "category": "objection",
     "text": "Jodi user jitchen na: Kono somosya nei Sir. Ekhon emon games achhe jekhane jethar shujog beshi, bonus-o dite pari."},
    {"id": "bn_obj_cuttingdown",   "language": "bengali", "category": "objection",
     "text": "Jodi user khela taggachhen: Apnar shiddhanto shomman kori Sir. Amader kache kom risky games achhe je gulo nirpad o mozadar."},
    {"id": "bn_obj_notinterested", "language": "bengali", "category": "objection",
     "text": "Kono pressure nei Sir. Ami offer pathiye debo, jakhon ready hoben takhon use korun. Offer matro teen din-er jonyo."},
    {"id": "bn_faq_claim",         "language": "bengali", "category": "faq",
     "text": "Offer claim korte: Login korun, Promotions-e jan, Claim Offer click korun, Rs 100 deposit korun. Bonus tukhuni credit hoye jay."},
    {"id": "bn_faq_trust",         "language": "bengali", "category": "faq",
     "text": "ABC Games purnorupe licensed o secure. Transactions encrypted, data private. Crore taka payout kora hoyeche. Support 24/7."},
    {"id": "bn_closing",           "language": "bengali", "category": "closing",
     "text": "Ami ki apnake text ba email korte pari? Poreri baar kakhon visit korben? Kono pressure nei Sir, ami offer pathiye debo."},

    # ═══════════════════════════════════════════════════════════════════
    # TELUGU (from Outbound Telugu Reactivate script)
    # ═══════════════════════════════════════════════════════════════════
    {"id": "te_offer_deposit",     "language": "telugu",  "category": "offer",
     "text": "ABC Games Rs 500 varaku 100% deposit match bonus istundi. Rs 500 deposit chesthe Rs 500 extra vastundi — mottam Rs 1000 aadadam kosam."},
    {"id": "te_offer_freespin",    "language": "telugu",  "category": "offer",
     "text": "Mana kotta games lo free spins vastundi reactivation offer tho patu tirigi vasthunna users ki."},
    {"id": "te_offer_highroller",  "language": "telugu",  "category": "offer",
     "text": "Ee weekend ABC Games high roller event undi. Tirigi vasthunna users ki exclusive entry istamu."},
    {"id": "te_offer_urgency",     "language": "telugu",  "category": "offer",
     "text": "Ee offer kevalam moodu rojulu matrame untundi. Moodu rojula tarvata automatically expire avutundi."},
    {"id": "te_offer_eligibility", "language": "telugu",  "category": "offer",
     "text": "Offer chivari 30 rojulalo visit cheyyani users kosam. Bonus kosam minimum Rs 100 deposit cheyyali."},
    {"id": "te_obj_busy",          "language": "telugu",  "category": "objection",
     "text": "User busy ga unte: Nenu artham chesukuntunnanu Sir, life chaala busy avuthundi. Kani isari meeku oka special bonus istunnamu, malli adadam start cheyavachu."},
    {"id": "te_obj_notwinning",    "language": "telugu",  "category": "objection",
     "text": "User gelavadam ledu antey: No problem Sir. Ippudu chala games unnay vaatilo gelavadaniki chances ekkuva. Nenu meeku oka bonus istanu."},
    {"id": "te_obj_cuttingdown",   "language": "telugu",  "category": "objection",
     "text": "User aadadam taggistaanu antey: Memu me decision ki respect istunnamu Sir. Mana degara low-risk games unnay, avi thakkuva risky mariyu deposit limits limited, fun kuda safe kuda untay."},
    {"id": "te_obj_notinterested", "language": "telugu",  "category": "objection",
     "text": "No pressure Sir. Nenu meeku offer pampistanu, meeru ready ainappudu use cheyandi. Ee offer kevalam moodu rojulu."},
    {"id": "te_faq_claim",         "language": "telugu",  "category": "faq",
     "text": "Offer claim cheyyadam kosam: Login cheyyandi, Promotions ki vellandi, Claim Offer click cheyyandi, Rs 100 deposit cheyyandi. Bonus ventane credit avutundi."},
    {"id": "te_faq_trust",         "language": "telugu",  "category": "faq",
     "text": "ABC Games purtigaa licensed mattu secure. Transactions encrypted, data private. Crores payout chesaamu. Support 24/7 available."},
    {"id": "te_closing",           "language": "telugu",  "category": "closing",
     "text": "Nenu meeku text leka email pampagalanaa? Next time visit cheyadaniki eppudu convenient? No pressure Sir, offer pampistanu ready ainappudu use cheyandi."},
]

COLLECTION_NAME = "abc_games"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE     = 1536


async def ingest():
    print("Connecting to Qdrant...")
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )

    print(f"Embedding {len(DOCUMENTS)} documents...")
    texts    = [doc["text"] for doc in DOCUMENTS]
    response = await oai.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    vectors  = [item.embedding for item in response.data]

    points = [
        models.PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "text":     DOCUMENTS[i]["text"],
                "id":       DOCUMENTS[i]["id"],
                "category": DOCUMENTS[i]["category"],
                "language": DOCUMENTS[i]["language"],
            },
        )
        for i in range(len(DOCUMENTS))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\n✅ {len(points)} documents ingested!\n")

    langs = {}
    for doc in DOCUMENTS:
        langs.setdefault(doc["language"], []).append(f"  [{doc['category']}] {doc['id']}")
    for lang, items in langs.items():
        print(f"{lang.upper()} ({len(items)} docs):")
        for item in items:
            print(item)
        print()


if __name__ == "__main__":
    asyncio.run(ingest())
