"""
Run ONCE locally: python ingest.py
Populates Qdrant with Bengali + Hindi knowledge base.
"""
import os
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from openai import AsyncOpenAI

load_dotenv()

DOCUMENTS = [
    # ── HINDI Offers ──────────────────────────────────────────────────────────
    {
        "id": "hi_offer_deposit",
        "text": "ABC Games Rs 500 tak ka 100% deposit match bonus deta hai. Matlab Rs 500 deposit karo to Rs 500 extra milega — total Rs 1000 khelne ke liye.",
        "category": "offer", "language": "hindi"
    },
    {
        "id": "hi_offer_freespin",
        "text": "Wapas aane wale users ko naye games mein free spin milte hain reactivation offer ke saath.",
        "category": "offer", "language": "hindi"
    },
    {
        "id": "hi_offer_highroller",
        "text": "Is weekend ABC Games ka high roller event hai. Wapas aane wale users ko exclusive entry milegi.",
        "category": "offer", "language": "hindi"
    },
    {
        "id": "hi_offer_urgency",
        "text": "Yeh reactivation offer — 100% deposit match Rs 500 tak, free spin, high roller entry — sirf teen din ke liye valid hai. Teen din baad automatically expire ho jaata hai.",
        "category": "offer", "language": "hindi"
    },
    {
        "id": "hi_offer_eligibility",
        "text": "Offer un users ke liye hai jo last 30 din mein visit nahi kiye. Minimum Rs 100 deposit karna hoga bonus activate karne ke liye.",
        "category": "offer", "language": "hindi"
    },
    {
        "id": "hi_objection_busy",
        "text": "Agar user busy ho: Samajhti hoon Sir. Isliye ek special bonus laye hain taki aap fir se khelna shuru karein. Kya main offer details bhej sakti hoon?",
        "category": "objection", "language": "hindi"
    },
    {
        "id": "hi_objection_notwinning",
        "text": "Agar user jeet nahi raha tha: Koi baat nahi Sir. Ab aise games hain jinmein jeetne ke chances zyada hain, plus bonus bhi milega.",
        "category": "objection", "language": "hindi"
    },
    {
        "id": "hi_objection_cuttingdown",
        "text": "Agar user khel kam karna chahta ho: Hum aapke decision ki respect karte hain. Hamare paas kam risky games hain jo safe aur mazedar dono hain.",
        "category": "objection", "language": "hindi"
    },
    {
        "id": "hi_objection_notinterested",
        "text": "Agar user interested nahi ho: No pressure Sir. Main offer send kar dungi, jab ready hon tab use kar lijiyega. Offer sirf teen din ke liye hai.",
        "category": "objection", "language": "hindi"
    },
    {
        "id": "hi_faq_claim",
        "text": "Offer claim karne ke liye: Login karein, Promotions mein jayein, Claim Offer click karein, Rs 100 deposit karein. Bonus turant credit hota hai.",
        "category": "faq", "language": "hindi"
    },
    {
        "id": "hi_faq_trust",
        "text": "ABC Games fully licensed aur secure hai. Transactions encrypted hain, data private hai. Crores ki payouts ki hain. Support 24/7 available hai.",
        "category": "faq", "language": "hindi"
    },
    {
        "id": "hi_closing",
        "text": "Call close karte waqt: Kya main aapko text ya email kar sakti hoon? Agli baar kab visit karenge? Agar hesitant ho: No pressure, offer send kar dungi.",
        "category": "closing", "language": "hindi"
    },

    # ── BENGALI Offers ────────────────────────────────────────────────────────
    {
        "id": "bn_offer_deposit",
        "text": "ABC Games Rs 500 paryanta 100% deposit match bonus dey. Arthat Rs 500 deposit korle Rs 500 extra paben — mot Rs 1000 khelar jonyo.",
        "category": "offer", "language": "bengali"
    },
    {
        "id": "bn_offer_freespin",
        "text": "Phire aasha users-der noya games-e free spin milbe reactivation offer-er sathe.",
        "category": "offer", "language": "bengali"
    },
    {
        "id": "bn_offer_highroller",
        "text": "Ei weekend-e ABC Games-er high roller event achhe. Phire aasha users-der exclusive entry dewa hobe.",
        "category": "offer", "language": "bengali"
    },
    {
        "id": "bn_offer_urgency",
        "text": "Ei reactivation offer — 100% deposit match Rs 500 paryanta, free spin, high roller entry — matro tin din-er jonyo valid. Tin din por automatically expire hoye jabe.",
        "category": "offer", "language": "bengali"
    },
    {
        "id": "bn_offer_eligibility",
        "text": "Offer tar jonyo jo users sesh 30 din-e visit koren ni. Bonus activate korte minimum Rs 100 deposit korte hobe.",
        "category": "offer", "language": "bengali"
    },
    {
        "id": "bn_objection_busy",
        "text": "Jodi user byasto thaken: Bujhte parchhi Sir. Tai apnar jonyo ekta special bonus niye eshechhi jate abar khelar shuru korte paren.",
        "category": "objection", "language": "bengali"
    },
    {
        "id": "bn_objection_notwinning",
        "text": "Jodi user jitche na bolen: Kono somosya nei Sir. Ekhon emon games achhe jekhane jethar shujog beshi, sathe bonus-o paben.",
        "category": "objection", "language": "bengali"
    },
    {
        "id": "bn_objection_cuttingdown",
        "text": "Jodi user khela kom korte chai: Apnar siddhanto-ke shomman kori Sir. Amar kache kom risky games achhe je gulo nirpad ebong mozadar.",
        "category": "objection", "language": "bengali"
    },
    {
        "id": "bn_objection_notinterested",
        "text": "Jodi user interested na hon: Kono pressure nei Sir. Ami offer pathiye debo, jakhon ready hoben takhon byabohar korben. Offer matro tin din-er jonyo.",
        "category": "objection", "language": "bengali"
    },
    {
        "id": "bn_faq_claim",
        "text": "Offer claim korte: Login korun, Promotions-e jan, Claim Offer click korun, Rs 100 deposit korun. Bonus tukhuni credit hoye jay.",
        "category": "faq", "language": "bengali"
    },
    {
        "id": "bn_faq_trust",
        "text": "ABC Games purnorupe licensed ebong secure. Transactions encrypted, data private rakha hoy. Crore taka payout kora hoyeche. Support 24/7 available.",
        "category": "faq", "language": "bengali"
    },
    {
        "id": "bn_closing",
        "text": "Call shesh korte: Ami ki apnake text ba email korte pari? Porer baar kakhon visit korben? Jodi dwidhagrost hon: Kono pressure nei, offer pathiye debo.",
        "category": "closing", "language": "bengali"
    },
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
    print(f"Collection created")

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
    print(f"✅ {len(points)} documents ingested!")
    for doc in DOCUMENTS:
        print(f"  [{doc['language']}][{doc['category']}] {doc['id']}")


if __name__ == "__main__":
    asyncio.run(ingest())