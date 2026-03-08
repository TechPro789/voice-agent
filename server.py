import os
import asyncio
import uuid
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from livekit.api import (
    AccessToken, VideoGrants, LiveKitAPI,
    CreateAgentDispatchRequest, CreateRoomRequest,
    DeleteRoomRequest
)

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

LIVEKIT_URL    = os.getenv("LIVEKIT_URL", "").strip()
LIVEKIT_KEY    = os.getenv("LIVEKIT_API_KEY", "").strip()
LIVEKIT_SECRET = os.getenv("LIVEKIT_API_SECRET", "").strip()

LANGUAGE_AGENTS = {
    "bengali": "bengali-agent",
    "hindi":   "hindi-agent",
    "telugu":  "telugu-agent",
}


@app.get("/health")
async def health():
    return {
        "LIVEKIT_URL":    LIVEKIT_URL or "MISSING",
        "LIVEKIT_KEY":    (LIVEKIT_KEY[:6] + "...") if LIVEKIT_KEY else "MISSING",
        "LIVEKIT_SECRET": "set" if LIVEKIT_SECRET else "MISSING",
        "OPENAI_KEY":     "set" if os.getenv("OPENAI_API_KEY") else "MISSING",
        "SARVAM_KEY":     "set" if os.getenv("SARVAM_API_KEY") else "MISSING",
        "QDRANT_URL":     os.getenv("QDRANT_URL", "MISSING"),
    }


@app.get("/token")
async def get_token(language: str = Query(default="bengali")):
    if not LIVEKIT_KEY or not LIVEKIT_SECRET or not LIVEKIT_URL:
        return JSONResponse(status_code=500, content={"error": "Missing LiveKit env vars"})

    if language not in LANGUAGE_AGENTS:
        return {"error": f"Unknown language. Choose: bengali, hindi, telugu"}

    agent_name = LANGUAGE_AGENTS[language]

    # ✅ Unique room name per session — prevents dispatch accumulation
    room_name = f"{language}-{uuid.uuid4().hex[:8]}"

    async with LiveKitAPI(LIVEKIT_URL, LIVEKIT_KEY, LIVEKIT_SECRET) as lk:

        # Step 1: Create fresh room
        try:
            room = await lk.room.create_room(
                CreateRoomRequest(
                    name=room_name,
                    empty_timeout=300,
                    max_participants=10,
                )
            )
            print(f"✅ Room created: {room.name}")
        except Exception as e:
            print(f"⚠️ Room note: {e}")

        # Step 2: Small wait for propagation
        await asyncio.sleep(0.5)

        # Step 3: Dispatch exactly ONE agent
        try:
            await lk.agent_dispatch.create_dispatch(
                CreateAgentDispatchRequest(
                    agent_name=agent_name,
                    room=room_name,
                )
            )
            print(f"✅ Dispatched {agent_name} → {room_name}")
        except Exception as e:
            print(f"⚠️ Dispatch note: {e}")

    # Step 4: Generate token for this unique room
    token = (
        AccessToken(LIVEKIT_KEY, LIVEKIT_SECRET)
        .with_identity("manager-test")
        .with_name("Manager")
        .with_grants(VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        .to_jwt()
    )

    return {"token": token, "room": room_name}


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


app.mount("/static", StaticFiles(directory="."), name="static")