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
    # HINDI
    # ═══════════════════════════════════════════════════════════════════
    {"id": "hi_offer_deposit",      "language": "hindi",   "category": "offer",
     "text": "ABC Games Rs 500 tak ka 100% deposit match bonus deta hai. Rs 500 deposit karo to Rs 500 extra milega — total Rs 1000 khelne ke liye."},
    {"id": "hi_offer_freespin",     "language": "hindi",   "category": "offer",
     "text": "Wapas aane wale users ko naye games mein free spin milte hain reactivation offer ke saath."},
    {"id": "hi_offer_highroller",   "language": "hindi",   "category": "offer",
     "text": "Is weekend ABC Games ka high roller event hai. Wapas aane wale users ko exclusive entry milegi."},
    {"id": "hi_offer_urgency",      "language": "hindi",   "category": "offer",
     "text": "Yeh reactivation offer sirf teen din ke liye valid hai. Teen din baad automatically expire ho jaata hai."},
    {"id": "hi_offer_eligibility",  "language": "hindi",   "category": "offer",
     "text": "Offer un users ke liye hai jo last 30 din mein visit nahi kiye. Minimum Rs 100 deposit karna hoga."},
    {"id": "hi_obj_busy",           "language": "hindi",   "category": "objection",
     "text": "Agar user busy ho: Samajhti hoon Sir. Isliye ek special bonus laye hain taki fir se khelna shuru karein."},
    {"id": "hi_obj_notwinning",     "language": "hindi",   "category": "objection",
     "text": "Agar user jeet nahi raha tha: Koi baat nahi Sir. Ab aise games hain jinmein jeetne ke chances zyada hain."},
    {"id": "hi_obj_cuttingdown",    "language": "hindi",   "category": "objection",
     "text": "Agar user khel kam karna chahta ho: Hamare paas kam risky games hain jo safe aur mazedar dono hain."},
    {"id": "hi_obj_notinterested",  "language": "hindi",   "category": "objection",
     "text": "Agar user interested nahi ho: No pressure Sir. Offer send kar dungi, jab ready hon tab use karein. Sirf teen din valid hai."},
    {"id": "hi_faq_claim",          "language": "hindi",   "category": "faq",
     "text": "Offer claim karne ke liye: Login karein, Promotions mein jayein, Claim Offer click karein, Rs 100 deposit karein. Bonus turant credit hota hai."},
    {"id": "hi_faq_trust",          "language": "hindi",   "category": "faq",
     "text": "ABC Games fully licensed aur secure hai. Transactions encrypted hain. Crores ki payouts ki hain. Support 24/7 available hai."},
    {"id": "hi_closing",            "language": "hindi",   "category": "closing",
     "text": "Call close karte waqt: Kya main text ya email kar sakti hoon? Agar hesitant ho: No pressure, offer send kar dungi."},

    # ═══════════════════════════════════════════════════════════════════
    # BENGALI
    # ═══════════════════════════════════════════════════════════════════
    {"id": "bn_offer_deposit",      "language": "bengali", "category": "offer",
     "text": "ABC Games Rs 500 paryanta 100% deposit match bonus dey. Rs 500 deposit korle Rs 500 extra paben — mot Rs 1000 khelar jonyo."},
    {"id": "bn_offer_freespin",     "language": "bengali", "category": "offer",
     "text": "Phire aasha users-der noya games-e free spin milbe reactivation offer-er sathe."},
    {"id": "bn_offer_highroller",   "language": "bengali", "category": "offer",
     "text": "Ei weekend-e ABC Games-er high roller event achhe. Phire aasha users-der exclusive entry dewa hobe."},
    {"id": "bn_offer_urgency",      "language": "bengali", "category": "offer",
     "text": "Ei reactivation offer matro tin din-er jonyo valid. Tin din por automatically expire hoye jabe."},
    {"id": "bn_offer_eligibility",  "language": "bengali", "category": "offer",
     "text": "Offer tar jonyo jo users sesh 30 din-e visit koren ni. Minimum Rs 100 deposit korte hobe."},
    {"id": "bn_obj_busy",           "language": "bengali", "category": "objection",
     "text": "Jodi user byasto thaken: Bujhte parchhi Sir. Tai apnar jonyo ekta special bonus niye eshechhi."},
    {"id": "bn_obj_notwinning",     "language": "bengali", "category": "objection",
     "text": "Jodi user jitche na bolen: Kono somosya nei Sir. Ekhon emon games achhe jekhane jethar shujog beshi."},
    {"id": "bn_obj_cuttingdown",    "language": "bengali", "category": "objection",
     "text": "Jodi user khela kom korte chai: Amar kache kom risky games achhe je gulo nirpad ebong mozadar."},
    {"id": "bn_obj_notinterested",  "language": "bengali", "category": "objection",
     "text": "Jodi user interested na hon: Kono pressure nei Sir. Ami offer pathiye debo, jakhon ready hoben takhon byabohar korben."},
    {"id": "bn_faq_claim",          "language": "bengali", "category": "faq",
     "text": "Offer claim korte: Login korun, Promotions-e jan, Claim Offer click korun, Rs 100 deposit korun. Bonus tukhuni credit hoye jay."},
    {"id": "bn_faq_trust",          "language": "bengali", "category": "faq",
     "text": "ABC Games purnorupe licensed ebong secure. Transactions encrypted. Crore taka payout kora hoyeche. Support 24/7 available."},
    {"id": "bn_closing",            "language": "bengali", "category": "closing",
     "text": "Call shesh korte: Ami ki apnake text ba email korte pari? Jodi dwidhagrost hon: Kono pressure nei, offer pathiye debo."},

    # ═══════════════════════════════════════════════════════════════════
    # TELUGU
    # ═══════════════════════════════════════════════════════════════════
    {"id": "te_offer_deposit",      "language": "telugu",  "category": "offer",
     "text": "ABC Games Rs 500 varaku 100% deposit match bonus istundi. Rs 500 deposit chesthe Rs 500 extra vastundi — mottam Rs 1000 aadadam kosam."},
    {"id": "te_offer_freespin",     "language": "telugu",  "category": "offer",
     "text": "Tirigi vasthunna users ki kotta games lo free spin vastundi reactivation offer tho patu."},
    {"id": "te_offer_highroller",   "language": "telugu",  "category": "offer",
     "text": "Ee weekend ABC Games high roller event undi. Tirigi vasthunna users ki exclusive entry istamu."},
    {"id": "te_offer_urgency",      "language": "telugu",  "category": "offer",
     "text": "Ee reactivation offer kevalam mudu rojulu valid. Mudu rojula tarvata automatically expire avutundi."},
    {"id": "te_offer_eligibility",  "language": "telugu",  "category": "offer",
     "text": "Offer chivari 30 rojulalo visit cheyyani users kosam. Bonus activate cheyyadam kosam minimum Rs 100 deposit cheyyali."},
    {"id": "te_obj_busy",           "language": "telugu",  "category": "objection",
     "text": "User busy ga unte: Artham avutundi Sir. Anduke mee kosam oka special bonus teccham, malli aadadam moudaladam kosam."},
    {"id": "te_obj_notwinning",     "language": "telugu",  "category": "objection",
     "text": "User gellavadam ledu antey: Parvaledhu Sir. Ippudu gellavadam ki ekkuva avakashaalu unna games unnay, bonus kuda vastundi."},
    {"id": "te_obj_cuttingdown",    "language": "telugu",  "category": "objection",
     "text": "User aadadam taggistaanu antey: Mee nirnayanni gouravastaamu Sir. Takkuva risky games unnay, avi surakshitanga mattu saredaga untayi."},
    {"id": "te_obj_notinterested",  "language": "telugu",  "category": "objection",
     "text": "User interested kaadu antey: Pressure ledu Sir. Offer pathistaanu, ready ayinapudu vadukondanki. Kevalam mudu rojulu valid."},
    {"id": "te_faq_claim",          "language": "telugu",  "category": "faq",
     "text": "Offer claim cheyyadam kosam: Login cheyyandi, Promotions ki vellandi, Claim Offer click cheyyandi, Rs 100 deposit cheyyandi. Bonus ventane credit avutundi."},
    {"id": "te_faq_trust",          "language": "telugu",  "category": "faq",
     "text": "ABC Games purtigaa licensed mattu secure. Transactions encrypted, data private ga unchutaamu. Crores payout chesaamu. Support 24/7 available."},
    {"id": "te_closing",            "language": "telugu",  "category": "closing",
     "text": "Call close cheyyadaniki: Mee ki text leda email cheyavacha? Sandehanga unte: Pressure ledu Sir, offer pathistaanu ready ayinapudu vadukondanki."},
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
    print(f"\n✅ {len(points)} documents ingested!")

    langs = {}
    for doc in DOCUMENTS:
        langs.setdefault(doc["language"], []).append(f"[{doc['category']}] {doc['id']}")
    for lang, items in langs.items():
        print(f"\n  {lang.upper()} ({len(items)} docs):")
        for item in items:
            print(f"    {item}")


if __name__ == "__main__":
    asyncio.run(ingest())