"""
Run this ONCE to populate Qdrant with your knowledge base.
Usage: python ingest.py
"""
import os
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from openai import AsyncOpenAI

load_dotenv()

# ─── Your ABC Games Knowledge Base ────────────────────────────────────────────
# Add / edit documents here. Each dict = one chunk stored in Qdrant.
DOCUMENTS = [
    # Offers
    {
        "id": "offer_deposit_match",
        "text": "ABC Games offers a 100% deposit match bonus up to Rs 500 for returning users. This means if a user deposits Rs 500, they get Rs 500 extra — total Rs 1000 to play with.",
        "category": "offer"
    },
    {
        "id": "offer_free_spins",
        "text": "Returning users also receive free spins as part of the reactivation offer. Free spins can be used on selected slot games on the ABC Games platform.",
        "category": "offer"
    },
    {
        "id": "offer_urgency",
        "text": "The reactivation offer — 100% deposit match up to Rs 500 plus free spins — is valid for only 3 days from the time the user is contacted. After 3 days the offer expires automatically.",
        "category": "offer"
    },
    {
        "id": "offer_eligibility",
        "text": "The offer is available to users who have not visited ABC Games in the last 30 days. Users must make a minimum deposit of Rs 100 to activate the bonus.",
        "category": "offer"
    },

    # Games
    {
        "id": "games_catalogue",
        "text": "ABC Games offers a wide range of games including cricket betting, IPL fantasy, teen patti, rummy, slots, and live casino games. All games are available 24/7 on mobile and desktop.",
        "category": "games"
    },
    {
        "id": "games_payments",
        "text": "ABC Games supports deposits and withdrawals via UPI, Paytm, PhonePe, Google Pay, net banking, and major debit/credit cards. Withdrawals are processed within 24 hours.",
        "category": "payments"
    },

    # FAQ / Objections
    {
        "id": "faq_safe",
        "text": "ABC Games is a fully licensed and secure gaming platform. All transactions are encrypted and user data is kept private. Customer support is available 24/7 via chat and phone.",
        "category": "faq"
    },
    {
        "id": "faq_how_to_claim",
        "text": "To claim the reactivation offer: 1) Log in to ABC Games, 2) Go to the Promotions section, 3) Click Claim Offer, 4) Make a deposit of at least Rs 100. The bonus is credited instantly.",
        "category": "faq"
    },
    {
        "id": "faq_wagering",
        "text": "The deposit match bonus has a 5x wagering requirement before withdrawal. For example, if you receive Rs 500 bonus, you must wager Rs 2500 before withdrawing the bonus amount.",
        "category": "faq"
    },
    {
        "id": "faq_not_interested",
        "text": "If the user says they are not interested or too busy, acknowledge politely, mention the offer expires in 3 days, and ask if it's okay to follow up later. Never be pushy.",
        "category": "objection_handling"
    },
    {
        "id": "faq_trust_issues",
        "text": "If the user is worried about money safety, assure them ABC Games uses bank-level encryption, is licensed, and has paid out crores to winners. Offer to send details via WhatsApp.",
        "category": "objection_handling"
    },
]

COLLECTION_NAME = "abc_games"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE     = 1536  # text-embedding-3-small output size


async def ingest():
    print("🔌 Connecting to Qdrant...")
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Create or recreate collection
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"♻️  Collection '{COLLECTION_NAME}' exists — recreating...")
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created")

    # Embed and upsert all documents
    print(f"📥 Embedding {len(DOCUMENTS)} documents...")
    texts = [doc["text"] for doc in DOCUMENTS]

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
            },
        )
        for i in range(len(DOCUMENTS))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ {len(points)} documents ingested into Qdrant!")
    print("\n📋 Ingested categories:")
    for doc in DOCUMENTS:
        print(f"  [{doc['category']}] {doc['id']}")


if __name__ == "__main__":
    asyncio.run(ingest())
