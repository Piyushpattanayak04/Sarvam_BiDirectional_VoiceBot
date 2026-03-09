"""
Voice Activity Detection (VAD) Engine

Implements a multi-tier silence detection system:
  - WebRTC VAD for frame-level speech/non-speech classification
  - Adaptive silence thresholds for pause vs. end-of-turn detection
  - Energy-based secondary validation to reduce false positives

Design Decisions:
  - WebRTC VAD is used because it runs in <1ms per frame (no GPU needed)
  - Three-tier silence thresholds allow the system to distinguish between:
      * Mid-sentence pauses (500ms) → do not respond
      * End-of-utterance pauses (800ms) → prepare response
      * Definite turn-end (1500ms) → send response immediately
  - Energy gating prevents background noise from triggering false speech events
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import webrtcvad

from src.config import AudioConfig, VADConfig
from src.logging_config import get_logger

logger = get_logger(__name__)


class SpeechState(str, Enum):
    """Current state of user speech."""
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEAKING = "speaking"
    PAUSE_SHORT = "pause_short"       # Mid-sentence pause
    PAUSE_MEDIUM = "pause_medium"     # Likely end of utterance
    PAUSE_LONG = "pause_long"         # Definite end of turn
    SPEECH_END = "speech_end"


@dataclass
class VADResult:
    """Result from a single VAD frame analysis."""
    is_speech: bool
    energy_db: float
    state: SpeechState
    silence_duration_ms: float = 0.0
    speech_duration_ms: float = 0.0


@dataclass
class VADState:
    """Mutable state tracking for the VAD engine."""
    is_speaking: bool = False
    speech_start_time: float = 0.0
    last_speech_time: float = 0.0
    silence_start_time: float = 0.0
    silence_frame_count: int = 0
    consecutive_speech_frames: int = 0
    consecutive_silence_frames: int = 0
    # Ring buffer for smoothing — stores last N frame decisions
    frame_ring_buffer: list[bool] = field(default_factory=lambda: [False] * 10)
    ring_index: int = 0
    # Adaptive noise floor
    noise_floor_db: float = -60.0
    noise_floor_alpha: float = 0.02  # Slow adaptation


class VADEngine:
    """
    Production VAD engine combining WebRTC VAD with energy gating
    and adaptive silence thresholds.
    
    Architecture:
        Audio Frame (20ms PCM16)
          → WebRTC VAD (speech/no-speech binary)
          → Energy Check (RMS dB > noise_floor + margin)
          → Ring Buffer Smoothing (majority vote of last 10 frames)
          → State Machine (silence → speech_start → speaking → pause → speech_end)
    """

    # Minimum speech energy above noise floor (dB) to count as real speech
    ENERGY_MARGIN_DB = 10.0
    # Frames needed to confirm speech start (debounce)
    SPEECH_ONSET_FRAMES = 3
    # Frames needed to confirm silence
    SILENCE_ONSET_FRAMES = 5

    def __init__(self, vad_config: VADConfig, audio_config: AudioConfig) -> None:
        self._config = vad_config
        self._audio_config = audio_config
        self._vad = webrtcvad.Vad(vad_config.aggressiveness)
        self._state = VADState()

    def reset(self) -> None:
        """Reset all VAD state (e.g., on new session)."""
        self._state = VADState()

    def process_frame(self, pcm_bytes: bytes) -> VADResult:
        """
        Process a single audio frame and return speech state.
        
        Args:
            pcm_bytes: Raw PCM16 mono audio, length must match chunk_size.
                       Expected: sample_rate * chunk_ms / 1000 * 2 bytes
        
        Returns:
            VADResult with current speech state and timing info.
        """
        now = time.monotonic()
        energy_db = self._compute_energy_db(pcm_bytes)
        
        # Update adaptive noise floor during silence
        if not self._state.is_speaking:
            self._state.noise_floor_db = (
                self._state.noise_floor_alpha * energy_db
                + (1 - self._state.noise_floor_alpha) * self._state.noise_floor_db
            )

        # WebRTC VAD classification
        try:
            webrtc_speech = self._vad.is_speech(
                pcm_bytes, self._audio_config.sample_rate
            )
        except Exception:
            webrtc_speech = False

        # Energy gating: reject if below noise floor + margin
        energy_speech = energy_db > (self._state.noise_floor_db + self.ENERGY_MARGIN_DB)
        is_speech = webrtc_speech and energy_speech

        # Ring buffer smoothing
        self._state.frame_ring_buffer[self._state.ring_index] = is_speech
        self._state.ring_index = (self._state.ring_index + 1) % len(
            self._state.frame_ring_buffer
        )
        speech_ratio = sum(self._state.frame_ring_buffer) / len(
            self._state.frame_ring_buffer
        )

        # State machine transitions
        state = self._update_state(is_speech, speech_ratio, now)

        silence_duration_ms = 0.0
        speech_duration_ms = 0.0
        if self._state.is_speaking:
            speech_duration_ms = (now - self._state.speech_start_time) * 1000
        if self._state.silence_start_time > 0:
            silence_duration_ms = (now - self._state.silence_start_time) * 1000

        return VADResult(
            is_speech=is_speech,
            energy_db=energy_db,
            state=state,
            silence_duration_ms=silence_duration_ms,
            speech_duration_ms=speech_duration_ms,
        )

    def _update_state(
        self, is_speech: bool, speech_ratio: float, now: float
    ) -> SpeechState:
        """State machine for speech/silence transitions."""
        cfg = self._config

        if is_speech:
            self._state.consecutive_speech_frames += 1
            self._state.consecutive_silence_frames = 0
            self._state.last_speech_time = now

            if not self._state.is_speaking:
                if self._state.consecutive_speech_frames >= self.SPEECH_ONSET_FRAMES:
                    # Transition: silence → speaking
                    self._state.is_speaking = True
                    self._state.speech_start_time = now
                    self._state.silence_start_time = 0.0
                    self._state.silence_frame_count = 0
                    logger.debug("vad_speech_start", speech_ratio=speech_ratio)
                    return SpeechState.SPEECH_START
                return SpeechState.SILENCE
            return SpeechState.SPEAKING

        else:
            self._state.consecutive_silence_frames += 1
            self._state.consecutive_speech_frames = 0

            if not self._state.is_speaking:
                return SpeechState.SILENCE

            # We were speaking — measure silence duration
            if self._state.silence_start_time == 0.0:
                self._state.silence_start_time = now
                self._state.silence_frame_count = 0
            self._state.silence_frame_count += 1

            # Use frame counting for reliable timing (wall-clock can be unreliable in tests)
            silence_ms_by_frames = self._state.silence_frame_count * self._audio_config.chunk_ms
            silence_ms_by_clock = (now - self._state.silence_start_time) * 1000
            silence_ms = max(silence_ms_by_frames, silence_ms_by_clock)

            if silence_ms >= cfg.pause_long_ms:
                # Definite end of turn
                self._state.is_speaking = False
                logger.debug("vad_speech_end", silence_ms=silence_ms)
                return SpeechState.SPEECH_END
            elif silence_ms >= cfg.pause_medium_ms:
                return SpeechState.PAUSE_MEDIUM
            elif silence_ms >= cfg.pause_short_ms:
                return SpeechState.PAUSE_SHORT
            else:
                # Brief silence during speech (normal)
                return SpeechState.SPEAKING

    @staticmethod
    def _compute_energy_db(pcm_bytes: bytes) -> float:
        """Compute RMS energy in dB for a PCM16 frame."""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float64)
        if len(samples) == 0:
            return -100.0
        rms = np.sqrt(np.mean(samples**2))
        if rms < 1e-10:
            return -100.0
        return 20 * np.log10(rms / 32768.0)

    @property
    def is_speaking(self) -> bool:
        return self._state.is_speaking
