# Sarvam BD VoiceBot

A real-time bidirectional voice banking assistant powered by **Sarvam AI**.  
Supports English, Tamil, Telugu, and Gujarati voice interactions for banking tasks like blocking cards, getting bank statements, and loan enquiries.

---

## Prerequisites

Before you begin, make sure you have:

- **Python 3.10+** installed ([python.org](https://www.python.org/downloads/))
- A **Sarvam AI API key** — sign up at [sarvam.ai](https://www.sarvam.ai/) and get your API key from the dashboard
- *(Optional)* An **OpenAI API key** — only needed if you want LLM-based intent fallback (the rule-based detector works fine without it)
- *(Optional)* A **LiveKit Cloud account** — only needed if you want WebRTC-based transport instead of plain WebSocket

---

## Setup Tasks (Step by Step)

### 1. Clone / Open the Project

```bash
cd d:\Internship\Sarvam_BD_VoiceBot
```

### 2. Create & Activate Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Create Your `.env` File

Copy the example and fill in your keys:

```powershell
Copy-Item .env.example .env
```

Then open `.env` and set these **required** values:

| Variable | Required? | Where to get it |
|----------|-----------|-----------------|
| `SARVAM_API_KEY` | **Yes** | [Sarvam AI Dashboard](https://www.sarvam.ai/) |
| `OPENAI_API_KEY` | No | [OpenAI Platform](https://platform.openai.com/) — for LLM fallback |
| `LIVEKIT_URL` | No | [LiveKit Cloud](https://livekit.io/) — for WebRTC mode |
| `LIVEKIT_API_KEY` | No | LiveKit project settings |
| `LIVEKIT_API_SECRET` | No | LiveKit project settings |

### 5. Run the Server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
uvicorn src.main:app --host 0.0.0.0 --port 8000 --log-level info
```

The server starts at `http://localhost:8000`. Verify with:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "active_sessions": 0, "max_sessions": 10}
```

### 6. Run Tests

```bash
pytest tests/ -v
```

---

## How It Works

### Architecture

```
Client (Mic)                          Server
    │                                   │
    │── PCM audio frames (WebSocket) ──▶│
    │                                   ├── VAD (voice activity detection)
    │                                   ├── Sarvam STT (speech-to-text)
    │                                   ├── NLU (intent + entity extraction)
    │                                   ├── Dialogue Manager (state machine)
    │                                   ├── Banking API (simulated)
    │                                   ├── Sarvam TTS (text-to-speech)
    │◀── TTS audio chunks (WebSocket) ──┤
    │                                   │
```

### WebSocket Protocol

Connect to `ws://localhost:8000/ws/voice`

**Client → Server:**
- **Binary frames**: Raw PCM16 audio (16kHz, mono, 20ms chunks = 640 bytes)
- **JSON text frames**:
  - `{"type": "config", "language": "ta-IN"}` — switch language
  - `{"type": "end"}` — end session

**Server → Client:**
- **Binary frames**: TTS audio chunks (PCM16)
- **JSON text frames**:
  - `{"type": "session_start", "session_id": "...", "config": {...}}`
  - `{"type": "state", "state": "listening"}`
  - `{"type": "ping"}` — keepalive

### Supported Banking Intents

| Intent | Example Phrases | Entities |
|--------|----------------|----------|
| Block Card | "I lost my debit card", "கார்டை பிளாக் செய்" | `card_type` (debit/credit) |
| Bank Statement | "Statement for last 30 days", "30 రోజుల స్టేట్‌మెంట్" | `days` (number) |
| Loan Enquiry | "Tell me about home loans", "હોમ લોન વિશે જણાવો" | `loan_type` (personal/home/car/education/business) |

### Supported Languages

| Language | Code | STT | TTS | Intent Detection |
|----------|------|-----|-----|-----------------|
| English | en-IN | ✅ | ✅ | ✅ |
| Tamil | ta-IN | ✅ | ✅ | ✅ |
| Telugu | te-IN | ✅ | ✅ | ✅ |
| Gujarati | gu-IN | ✅ | ✅ | ✅ |

---

## Project Structure

```
Sarvam_BD_VoiceBot/
├── src/
│   ├── main.py                          # Entry point
│   ├── config.py                        # Settings (from .env)
│   ├── logging_config.py                # Structured logging
│   ├── audio/
│   │   ├── vad_engine.py                # Voice activity detection (WebRTC VAD)
│   │   ├── interruption_handler.py      # Barge-in / backchannel detection
│   │   └── stream_manager.py            # Audio I/O orchestration
│   ├── nlu/
│   │   ├── intent_detector.py           # Rule-based + LLM intent detection
│   │   ├── entity_extractor.py          # Multilingual entity extraction
│   │   └── pipeline.py                  # NLU pipeline coordinator
│   ├── conversation/
│   │   ├── state.py                     # Dialogue states & context
│   │   ├── dialogue_manager.py          # Conversation state machine
│   │   └── session_store.py             # In-memory session storage
│   ├── banking/
│   │   └── api_client.py                # Simulated banking APIs
│   └── server/
│       ├── websocket_server.py          # FastAPI WebSocket server
│       ├── voice_session.py             # Per-user session orchestrator
│       └── livekit_agent.py             # LiveKit agent (alternative mode)
├── tests/
│   ├── conftest.py                      # Shared fixtures
│   ├── test_vad_engine.py               # VAD tests
│   ├── test_nlu.py                      # Intent & entity tests
│   ├── test_dialogue.py                 # Conversation flow tests
│   ├── test_banking_api.py              # Banking API tests
│   └── test_interruption.py             # Interruption handling tests
├── pyproject.toml                       # Dependencies & tool config
├── Dockerfile                           # Container build
├── docker-compose.yml                   # Single-container deployment
├── .env.example                         # Environment template
└── .gitignore
```

---

## Alternative: Run with Docker

```bash
# Build and run
docker-compose up --build

# Or just Docker directly
docker build -t voicebot .
docker run -p 8000:8000 --env-file .env voicebot
```

---

## Alternative: LiveKit Mode

If you have a LiveKit Cloud account, you can run the agent in WebRTC mode (better audio quality, built-in echo cancellation):

```bash
python -m src.server.livekit_agent dev
```

Make sure `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` are set in your `.env`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `SARVAM_API_KEY` error on startup | Make sure `.env` file exists and has your key |
| `webrtcvad` build fails | Install C build tools: `pip install setuptools wheel` and on Windows ensure Visual C++ Build Tools are installed |
| WebSocket connection refused | Check the server is running on port 8000 |
| No audio response | Verify your Sarvam API key is valid and has credits |
| Tests fail with import errors | Make sure you installed with `pip install -e ".[dev]"` |

---

## Quick Reference Commands

```bash
# Start server (with hot reload)
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_nlu.py -v

# Check code style
ruff check src/ tests/

# Type check
mypy src/
```
