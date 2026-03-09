"""
LiveKit Voice Agent Integration

Production-grade voice agent using LiveKit + Sarvam AI plugins.
This is the recommended approach for real-time bidirectional voice
when deploying to production with WebRTC infrastructure.

LiveKit handles:
  - WebRTC media transport (low-latency audio)
  - Room management and participant tracking
  - Network quality adaptation
  - Echo cancellation and noise suppression

Sarvam plugins handle:
  - Streaming STT (Saaras v3)
  - Streaming TTS (Bulbul v3)
  - VAD and turn detection

This agent adds:
  - Banking-specific conversation logic
  - Intent detection and entity extraction
  - Dialogue state management
  - Banking API integration
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv()

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, sarvam

from src.logging_config import get_logger

logger = get_logger(__name__)


BANKING_SYSTEM_PROMPT = """You are a professional banking voice assistant for an Indian bank.
You help customers with:
1. Blocking lost/stolen cards (debit/credit)
2. Generating bank statements for a specific period
3. Loan enquiries (personal, home, car, education, business)

Important guidelines:
- Be polite, professional, and empathetic
- Speak concisely — you are a voice agent, not a chatbot
- Confirm important actions before executing (like blocking a card)
- Ask for missing information naturally
- Support English, Tamil, Telugu, and Gujarati
- If the user speaks in a regional language, respond in the same language
- For bank statements, always ask how many days if not specified
- For card blocking, confirm the card type and ask for confirmation
- Never share sensitive information like full card numbers
- Keep responses under 3 sentences for natural voice conversation

Example flows:
User: "I lost my debit card"
You: "I'm sorry to hear that. I'll block your debit card right away. Can you confirm you'd like me to proceed?"

User: "I need my bank statement"
You: "Sure! For how many days would you like the statement?"

User: "Tell me about personal loans"
You: "We offer personal loans starting from 10.5% annual interest. Would you like me to check your eligibility?"
"""


class BankingVoiceAgent(Agent):
    """
    LiveKit-based banking voice agent with Sarvam AI plugins.
    
    This is the simplest production path:
    - LiveKit handles all audio transport and WebRTC
    - Sarvam STT/TTS plugins handle speech processing
    - OpenAI/Sarvam LLM handles conversation logic
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=BANKING_SYSTEM_PROMPT,
            # Sarvam Saaras v3 — Streaming STT
            stt=sarvam.STT(
                language="unknown",  # Auto-detect language
                model="saaras:v3",
                mode="transcribe",
                flush_signal=True,   # Enable speech start/end events
            ),
            # LLM for conversation
            llm=openai.LLM(model="gpt-4o"),
            # Sarvam Bulbul v3 — Streaming TTS
            tts=sarvam.TTS(
                target_language_code="en-IN",
                model="bulbul:v3",
                speaker="shubh",
            ),
        )

    async def on_enter(self) -> None:
        """Called when user joins — agent starts the conversation."""
        self.session.generate_reply()


async def entrypoint(ctx: JobContext) -> None:
    """Main entry point — LiveKit calls this when a user connects."""
    logger.info("user_connected", room=ctx.room.name)

    session = AgentSession(
        turn_detection="stt",           # Use Sarvam STT for turn detection
        min_endpointing_delay=0.07,     # 70ms — matches Sarvam STT processing
    )

    await session.start(
        agent=BankingVoiceAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
