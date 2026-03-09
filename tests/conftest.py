"""Test configuration and shared fixtures."""

import asyncio
from unittest.mock import MagicMock

import pytest

from src.config import AppConfig, AudioConfig, SarvamConfig, VADConfig


@pytest.fixture
def vad_config() -> VADConfig:
    return VADConfig()


@pytest.fixture
def audio_config() -> AudioConfig:
    return AudioConfig()


@pytest.fixture
def sarvam_config() -> SarvamConfig:
    return SarvamConfig(api_key="test_key_not_real")  # type: ignore[call-arg]


@pytest.fixture
def app_config(sarvam_config: SarvamConfig) -> AppConfig:
    """App config with test defaults."""
    return AppConfig(sarvam=sarvam_config)  # type: ignore[call-arg]


@pytest.fixture
def silence_frame(audio_config: AudioConfig) -> bytes:
    """20ms of silence as PCM16."""
    return b"\x00\x00" * audio_config.frames_per_chunk


@pytest.fixture
def speech_frame(audio_config: AudioConfig) -> bytes:
    """20ms of simulated speech (sine wave) as PCM16."""
    import math
    import struct

    frames = audio_config.frames_per_chunk
    freq = 440  # Hz
    samples = []
    for i in range(frames):
        t = i / audio_config.sample_rate
        sample = int(16000 * math.sin(2 * math.pi * freq * t))
        samples.append(struct.pack("<h", max(-32768, min(32767, sample))))
    return b"".join(samples)
