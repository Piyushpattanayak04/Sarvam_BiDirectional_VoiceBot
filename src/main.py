"""
Application entry point.

Run with:
  # WebSocket server (custom implementation)
  uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

  # LiveKit agent (production-recommended)
  python -m src.server.livekit_agent dev
"""

from src.server.websocket_server import app

__all__ = ["app"]
