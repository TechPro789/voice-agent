# Setup Guide — RAG + Noise Cancellation

## 1. New Environment Variables

Add these to your `.env` and Render environment:

```env
QDRANT_URL=https://YOUR-CLUSTER.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

Get free Qdrant Cloud credentials at: https://cloud.qdrant.io
(Free tier: 1GB, no credit card needed)

## 2. Install new dependencies

```bash
pip install qdrant-client==1.13.3 livekit-plugins-silero
```

## 3. Ingest your knowledge base into Qdrant

Run this ONCE locally after setting env vars:

```bash
python ingest.py
```

You should see:
```
✅ Collection 'abc_games' created
✅ 11 documents ingested into Qdrant!
```

To add more knowledge — edit the DOCUMENTS list in ingest.py and re-run.

## 4. Deploy to Render

Add QDRANT_URL and QDRANT_API_KEY to both services
(voice-agent web + voice-agent-worker) in Render dashboard.

Push to GitHub — Render auto-deploys.

## What's New

### RAG (No more hallucination)
- Mia now queries Qdrant before every response
- Only verified offer details are injected into the LLM prompt
- If asked something not in the database, Mia says she'll check

### Silero VAD (Server-side noise filtering)
- Runs on the agent worker
- Filters out background noise and silence before STT
- Only real speech reaches Sarvam — reduces false transcriptions

### RNNoise (Browser-side noise cancellation)
- ML model runs in the browser (WebAssembly)
- Same technology used by Zoom and Discord
- Cleans mic audio before it's sent to LiveKit
- Falls back gracefully if WASM doesn't load
