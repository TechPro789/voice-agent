import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from livekit.api import AccessToken, VideoGrants, LiveKitAPI, CreateAgentDispatchRequest

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_SECRET = os.getenv("LIVEKIT_API_SECRET")

ROOM_NAME = "manager-room"


@app.get("/token")
async def get_token():

    if not LIVEKIT_URL or not LIVEKIT_KEY or not LIVEKIT_SECRET:
        return {"error": "Missing LiveKit environment variables"}

    # Dispatch agent
    async with LiveKitAPI(LIVEKIT_URL, LIVEKIT_KEY, LIVEKIT_SECRET) as lk:
        try:
            await lk.agent_dispatch.create_dispatch(
                CreateAgentDispatchRequest(
                    agent_name="",
                    room=ROOM_NAME
                )
            )
        except Exception as e:
            print("Dispatch warning:", e)

    grant = VideoGrants(
        room_join=True,
        room=ROOM_NAME,
        can_publish=True,
        can_subscribe=True
    )

    token = (
        AccessToken(LIVEKIT_KEY, LIVEKIT_SECRET)
        .with_identity("manager-test")
        .with_name("Manager")
        .with_grants(grant)
        .to_jwt()
    )

    return {"token": token}


# serve frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")