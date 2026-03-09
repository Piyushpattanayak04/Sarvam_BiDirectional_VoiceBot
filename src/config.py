"""
Application configuration loaded from environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class SarvamConfig(BaseSettings):
    """Sarvam AI API configuration."""

    api_key: str = Field("", alias="SARVAM_API_KEY")
    stt_model: str = "saaras:v3"
    stt_mode: str = "transcribe"
    tts_model: str = "bulbul:v3"
    tts_default_speaker: str = "shubh"
    llm_model: str = "sarvam-30b"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class LiveKitConfig(BaseSettings):
    """LiveKit configuration."""

    url: str = Field("", alias="LIVEKIT_URL")
    api_key: str = Field("", alias="LIVEKIT_API_KEY")
    api_secret: str = Field("", alias="LIVEKIT_API_SECRET")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class GroqConfig(BaseSettings):
    """Groq API configuration for LLM fallback intent classification."""

    api_key: str = Field("", alias="GROQ_API_KEY")
    llm_model: str = "llama-3.1-8b-instant"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class VADConfig(BaseSettings):
    """Voice Activity Detection configuration."""

    silence_threshold_ms: int = Field(800, alias="VAD_SILENCE_THRESHOLD_MS")
    speech_min_duration_ms: int = Field(250, alias="VAD_SPEECH_MIN_DURATION_MS")
    aggressiveness: int = Field(2, alias="VAD_AGGRESSIVENESS")  # 0-3, higher = more aggressive
    # Adaptive thresholds
    pause_short_ms: int = 500   # Short pause (mid-sentence)
    pause_medium_ms: int = 800  # Medium pause (likely end of utterance)
    pause_long_ms: int = 1500   # Long pause (definite end of turn)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


class AudioConfig(BaseSettings):
    """Audio pipeline configuration."""

    sample_rate: int = Field(16000, alias="AUDIO_SAMPLE_RATE")
    channels: int = Field(1, alias="AUDIO_CHANNELS")
    chunk_ms: int = Field(20, alias="AUDIO_CHUNK_MS")
    bit_depth: int = 16

    @property
    def chunk_size(self) -> int:
        """Bytes per audio chunk."""
        return int(self.sample_rate * self.channels * (self.bit_depth // 8) * self.chunk_ms / 1000)

    @property
    def frames_per_chunk(self) -> int:
        """PCM frames per chunk."""
        return int(self.sample_rate * self.chunk_ms / 1000)


class AppConfig(BaseSettings):
    """Root application configuration."""

    env: Environment = Field(Environment.DEVELOPMENT, alias="APP_ENV")
    port: int = Field(8000, alias="APP_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    max_concurrent_sessions: int = Field(10, alias="MAX_CONCURRENT_SESSIONS")
    response_timeout_ms: int = Field(800, alias="RESPONSE_TIMEOUT_MS")

    sarvam: SarvamConfig = SarvamConfig()  # type: ignore[call-arg]
    groq: GroqConfig = GroqConfig()  # type: ignore[call-arg]
    livekit: LiveKitConfig = LiveKitConfig()
    vad: VADConfig = VADConfig()
    audio: AudioConfig = AudioConfig()

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache
def get_config() -> AppConfig:
    """Singleton config instance."""
    return AppConfig()
