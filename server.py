import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from livekit.api import (
    AccessToken, VideoGrants, LiveKitAPI,
    CreateAgentDispatchRequest
)
from livekit.api.livekit_room import CreateRoomRequest

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

LIVEKIT_URL    = os.getenv("LIVEKIT_URL")
LIVEKIT_KEY    = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_SECRET = os.getenv("LIVEKIT_API_SECRET")

LANGUAGE_ROOMS = {
    "bengali": {"room": "bengali-room", "agent": "bengali-agent"},
    "hindi":   {"room": "hindi-room",   "agent": "hindi-agent"},
    "telugu":  {"room": "telugu-room",  "agent": "telugu-agent"},
}


@app.get("/token")
async def get_token(language: str = Query(default="bengali")):
    if language not in LANGUAGE_ROOMS:
        return {"error": f"Unknown language '{language}'. Choose: bengali, hindi, telugu"}

    cfg        = LANGUAGE_ROOMS[language]
    room_name  = cfg["room"]
    agent_name = cfg["agent"]

    async with LiveKitAPI(LIVEKIT_URL, LIVEKIT_KEY, LIVEKIT_SECRET) as lk:

        # ✅ Step 1: Create the room explicitly FIRST
        try:
            room = await lk.room.create_room(
                CreateRoomRequest(
                    name=room_name,
                    empty_timeout=300,   # close room after 5 mins if empty
                    max_participants=10,
                )
            )
            print(f"✅ Room created: {room.name} (sid={room.sid})")
        except Exception as e:
            print(f"⚠️ Room may already exist: {e}")

        # ✅ Step 2: Dispatch the agent AFTER room exists
        try:
            dispatch = await lk.agent_dispatch.create_dispatch(
                CreateAgentDispatchRequest(
                    agent_name=agent_name,
                    room=room_name,
                )
            )
            print(f"✅ Dispatched: {agent_name} → {room_name}")
        except Exception as e:
            print(f"⚠️ Dispatch note: {e}")

    # ✅ Step 3: Generate token
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


app.mount("/", StaticFiles(directory=".", html=True), name="static")