"""
WebSocket Server — Entry point for client connections.

Handles:
  - WebSocket upgrade and session management
  - Binary audio frame ingestion
  - JSON control messages (language change, end session)
  - TTS audio streaming back to client
  - Connection lifecycle (connect, disconnect, error recovery)
  - Concurrent session limits

Protocol:
  Client → Server:
    - Binary frames: Raw PCM16 audio (20ms chunks at 16kHz mono)
    - Text frames: JSON control messages
      {"type": "config", "language": "ta-IN"}
      {"type": "end"}
  
  Server → Client:
    - Binary frames: TTS audio (PCM16 or encoded)
    - Text frames: JSON events
      {"type": "transcript", "text": "...", "is_final": true}
      {"type": "response", "text": "..."}
      {"type": "state", "state": "listening"}
      {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_config
from src.conversation.session_store import InMemorySessionStore
from src.logging_config import get_logger, setup_logging
from src.server.voice_session import VoiceSession

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    setup_logging(config.log_level, json_output=config.env.value == "production")

    app = FastAPI(
        title="Sarvam BD VoiceBot",
        version="1.0.0",
        description="Bidirectional Voice Conversation Agent for Banking",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve test client HTML
    from fastapi.responses import FileResponse
    import os

    @app.get("/")
    async def serve_test_client():
        html_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_client.html")
        return FileResponse(os.path.abspath(html_path), media_type="text/html")

    # Shared session store (in-memory for hobby/small-scale use)
    session_store = InMemorySessionStore()

    # Active sessions tracking
    active_sessions: dict[str, VoiceSession] = {}

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "active_sessions": len(active_sessions),
            "max_sessions": config.max_concurrent_sessions,
        }

    @app.websocket("/ws/voice")
    async def voice_websocket(websocket: WebSocket):
        """
        Main WebSocket endpoint for bidirectional voice streaming.
        
        Connection flow:
          1. Client connects → server assigns session_id
          2. Server sends config acknowledgment
          3. Client streams audio frames (binary)
          4. Server streams TTS audio back (binary)
          5. Both sides can send control messages (JSON)
        """
        await websocket.accept()
        session_id = str(uuid.uuid4())[:12]

        logger.info("ws_connected", session_id=session_id)

        # Check concurrent session limit
        if len(active_sessions) >= config.max_concurrent_sessions:
            await websocket.send_json({
                "type": "error",
                "message": "Server at capacity. Please try again later.",
            })
            await websocket.close(code=1013)
            return

        # Create voice session
        session = VoiceSession(
            session_id=session_id,
            config=config,
            session_store=session_store,
        )
        active_sessions[session_id] = session

        try:
            await session.start()

            # Send session config to client
            await websocket.send_json({
                "type": "session_start",
                "session_id": session_id,
                "config": {
                    "sample_rate": config.audio.sample_rate,
                    "channels": config.audio.channels,
                    "chunk_ms": config.audio.chunk_ms,
                    "bit_depth": config.audio.bit_depth,
                },
            })

            # Run input and output loops concurrently
            await asyncio.gather(
                _input_loop(websocket, session),
                _output_loop(websocket, session),
            )

        except WebSocketDisconnect:
            logger.info("ws_disconnected", session_id=session_id)
        except Exception as e:
            # Filter out common disconnect-related errors
            err_msg = str(e)
            if "disconnect" in err_msg.lower() or "websocket.close" in err_msg.lower():
                logger.info("ws_disconnected", session_id=session_id)
            else:
                logger.error("ws_error", session_id=session_id, error=err_msg)
        finally:
            await session.stop()
            active_sessions.pop(session_id, None)
            logger.info(
                "session_cleanup",
                session_id=session_id,
                remaining_sessions=len(active_sessions),
            )

    return app


async def _input_loop(websocket: WebSocket, session: VoiceSession) -> None:
    """Receive audio frames and control messages from the client."""
    while True:
        try:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                # Binary audio frame
                await session.on_audio_frame(message["bytes"])

            elif "text" in message and message["text"]:
                # JSON control message
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "config":
                        language = data.get("language", "en-IN")
                        logger.info(
                            "language_changed",
                            session_id=session.session_id,
                            language=language,
                        )

                    elif msg_type == "end":
                        logger.info("session_end_requested", session_id=session.session_id)
                        break

                except json.JSONDecodeError:
                    logger.warning("invalid_control_message", session_id=session.session_id)

        except WebSocketDisconnect:
            break


async def _output_loop(websocket: WebSocket, session: VoiceSession) -> None:
    """Stream TTS audio and events back to the client."""
    while True:
        try:
            chunk = await asyncio.wait_for(
                session.audio_manager.tts_output_queue.get(),
                timeout=60.0,
            )

            if chunk is None:
                # All audio for this turn has been sent to the client.
                # Only NOW deactivate barge-in detection so the user can
                # interrupt during the full playback window (not just during
                # the server-side TTS download).
                session.audio_manager.interruption_handler.set_assistant_speaking(False)
                await websocket.send_json({"type": "state", "state": "listening"})
            else:
                # Send binary audio chunk
                await websocket.send_bytes(chunk)

        except asyncio.TimeoutError:
            # Send keepalive
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break
        except (WebSocketDisconnect, RuntimeError):
            # Client disconnected
            break
        except Exception as e:
            err_msg = str(e)
            if "disconnect" in err_msg.lower() or "websocket" in err_msg.lower():
                break
            logger.error("output_loop_error", error=err_msg)
            break


# ──────────────────────────────────────────────────────────────────────
# Application entry point
# ──────────────────────────────────────────────────────────────────────

app = create_app()
