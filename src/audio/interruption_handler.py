"""
Interruption Handler

Manages the complex interaction between user speech and assistant TTS playback.

Key scenarios handled:
  1. User starts speaking while assistant is playing TTS → stop TTS, capture user speech
  2. User interrupts with a short interjection ("yes", "ok") → treat as confirmation
  3. User interrupts with a new utterance → cancel current response, process new input
  4. Multiple rapid interruptions → debounce to avoid thrashing

Design Decisions:
  - Interruption detection runs on a separate async task monitoring VAD state
  - TTS cancellation is cooperative: we set a flag and the TTS streamer checks it
  - A minimum speech duration (250ms) prevents noise from triggering false interruptions
  - After interruption, there's a 100ms cooldown before re-enabling TTS
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum

from src.audio.vad_engine import SpeechState, VADResult
from src.config import VADConfig
from src.logging_config import get_logger

logger = get_logger(__name__)


class InterruptionType(str, Enum):
    """Classification of user interruptions."""
    NONE = "none"
    BARGE_IN = "barge_in"            # User takes over conversation
    BACKCHANNEL = "backchannel"      # Short acknowledgment ("yes", "hmm")
    NOISE = "noise"                  # Not a real interruption


@dataclass
class InterruptionState:
    """Tracks the current interruption state."""
    assistant_speaking: bool = False
    tts_cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    last_interruption_time: float = 0.0
    interruption_count: int = 0
    cooldown_until: float = 0.0

    def __post_init__(self) -> None:
        # Event is "set" when TTS should be cancelled
        self.tts_cancel_event.clear()


class InterruptionHandler:
    """
    Coordinates between user VAD events and assistant TTS playback.

    Flow:
        VADEngine emits SpeechState
          → InterruptionHandler evaluates context
          → If assistant_speaking AND user_speech detected:
              → Classify interruption type
              → If barge_in: Set tts_cancel_event, notify pipeline
              → If backchannel: Log but continue TTS
              → If noise: Ignore
    """

    # Minimum speech duration to count as real interruption (ms).
    # 200ms avoids speaker echo false-triggers while still being responsive.
    MIN_INTERRUPTION_DURATION_MS = 200
    # Cooldown between interruptions to prevent thrashing (ms)
    INTERRUPTION_COOLDOWN_MS = 100
    # Maximum speech duration for backchannel classification (ms).
    # Speech longer than this is always treated as a barge-in.
    BACKCHANNEL_MAX_DURATION_MS = 400

    def __init__(self, vad_config: VADConfig) -> None:
        self._config = vad_config
        self._state = InterruptionState()
        self._listeners: list[asyncio.Queue[InterruptionType]] = []

    def register_listener(self) -> asyncio.Queue[InterruptionType]:
        """Register a listener that will receive interruption events."""
        queue: asyncio.Queue[InterruptionType] = asyncio.Queue()
        self._listeners.append(queue)
        return queue

    def set_assistant_speaking(self, speaking: bool) -> None:
        """Called by TTS streamer when playback starts/stops."""
        self._state.assistant_speaking = speaking
        if not speaking:
            self._state.tts_cancel_event.clear()
        logger.debug("assistant_speaking_changed", speaking=speaking)

    @property
    def should_cancel_tts(self) -> bool:
        """Check if TTS should be cancelled (polled by TTS streamer)."""
        return self._state.tts_cancel_event.is_set()

    @property
    def tts_cancel_event(self) -> asyncio.Event:
        return self._state.tts_cancel_event

    async def on_vad_result(self, result: VADResult) -> InterruptionType:
        """
        Process a VAD result and determine if an interruption occurred.
        
        Should be called for every VAD frame during the session.
        """
        now = time.monotonic()

        # Not an interruption if assistant isn't speaking
        if not self._state.assistant_speaking:
            return InterruptionType.NONE

        # Check cooldown
        if now < self._state.cooldown_until:
            return InterruptionType.NONE

        # No speech detected
        if result.state in (SpeechState.SILENCE, SpeechState.SPEECH_END):
            return InterruptionType.NONE

        # Speech detected while assistant is speaking
        if result.state in (SpeechState.SPEECH_START, SpeechState.SPEAKING):
            interruption_type = self._classify_interruption(result)

            if interruption_type == InterruptionType.BARGE_IN:
                logger.info(
                    "interruption_detected",
                    type="barge_in",
                    speech_duration_ms=result.speech_duration_ms,
                    count=self._state.interruption_count + 1,
                )
                self._state.tts_cancel_event.set()
                self._state.last_interruption_time = now
                self._state.interruption_count += 1
                self._state.cooldown_until = now + self.INTERRUPTION_COOLDOWN_MS / 1000

                # Notify listeners
                for q in self._listeners:
                    await q.put(interruption_type)

                return InterruptionType.BARGE_IN

            elif interruption_type == InterruptionType.BACKCHANNEL:
                logger.debug("backchannel_detected", speech_ms=result.speech_duration_ms)
                return InterruptionType.BACKCHANNEL

            elif interruption_type == InterruptionType.NOISE:
                logger.debug("noise_ignored", speech_ms=result.speech_duration_ms)
                return InterruptionType.NOISE

        return InterruptionType.NONE

    def _classify_interruption(self, result: VADResult) -> InterruptionType:
        """
        Classify the type of interruption based on speech characteristics.
        
        Heuristics:
          - If speech < MIN_INTERRUPTION_DURATION_MS → noise
          - If speech < BACKCHANNEL_MAX_DURATION_MS → backchannel
          - Otherwise → barge_in (real interruption)
        """
        if result.speech_duration_ms < self.MIN_INTERRUPTION_DURATION_MS:
            return InterruptionType.NOISE

        if result.speech_duration_ms < self.BACKCHANNEL_MAX_DURATION_MS:
            # Could be backchannel, but if energy is adequate, treat as barge-in.
            # Threshold lowered to -30 dB so normal-volume speech always interrupts.
            if result.energy_db > -30.0:
                return InterruptionType.BARGE_IN
            return InterruptionType.BACKCHANNEL

        return InterruptionType.BARGE_IN

    def reset(self) -> None:
        """Reset interruption state for a new conversation."""
        self._state = InterruptionState()
        self._listeners.clear()
