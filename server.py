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


async def wait_for_room_sid(lk: LiveKitAPI, room_name: str, retries: int = 8, delay: float = 0.5) -> str:
    """
    Poll LiveKit until the room has a real SID.
    roomID being empty means the room isn't fully registered yet —
    dispatching before this causes the agent to silently fail.
    """
    for attempt in range(retries):
        try:
            rooms = await lk.room.list_rooms(names=[room_name])
            if rooms and rooms[0].sid:
                print(f"✅ Room SID confirmed: {rooms[0].sid} (attempt {attempt + 1})")
                return rooms[0].sid
        except Exception as e:
            print(f"⚠️ Room poll error (attempt {attempt + 1}): {e}")
        await asyncio.sleep(delay)

    print(f"⚠️ Room SID never confirmed after {retries} attempts — dispatching anyway")
    return ""


@app.get("/token")
async def get_token(language: str = Query(default="bengali")):
    if not LIVEKIT_KEY or not LIVEKIT_SECRET or not LIVEKIT_URL:
        return JSONResponse(status_code=500, content={"error": "Missing LiveKit env vars"})

    if language not in LANGUAGE_AGENTS:
        return JSONResponse(status_code=400, content={"error": "Unknown language. Choose: bengali, hindi, telugu"})

    agent_name = LANGUAGE_AGENTS[language]
    room_name  = f"{language}-{uuid.uuid4().hex[:8]}"

    async with LiveKitAPI(LIVEKIT_URL, LIVEKIT_KEY, LIVEKIT_SECRET) as lk:

        # Step 1: Create the room
        try:
            room = await lk.room.create_room(
                CreateRoomRequest(
                    name=room_name,
                    empty_timeout=300,
                    max_participants=10,
                )
            )
            print(f"✅ Room created: {room.name} (sid={room.sid!r})")
        except Exception as e:
            print(f"⚠️ Room create error: {e}")
            return JSONResponse(status_code=500, content={"error": f"Room creation failed: {e}"})

        # Step 2: Wait until LiveKit Cloud confirms the room has a real SID
        # ✅ FIX: 0.5s flat sleep was not enough — room SID was empty → dispatch silently failed
        await wait_for_room_sid(lk, room_name)

        # Step 3: Dispatch exactly ONE agent
        try:
            dispatch = await lk.agent_dispatch.create_dispatch(
                CreateAgentDispatchRequest(
                    agent_name=agent_name,
                    room=room_name,
                )
            )
            print(f"✅ Dispatched {agent_name} → {room_name} (dispatch_id={dispatch.id!r})")
        except Exception as e:
            print(f"❌ Dispatch failed: {e}")
            return JSONResponse(status_code=500, content={"error": f"Agent dispatch failed: {e}"})

    # Step 4: Generate token for this room
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