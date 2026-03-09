"""Tests for the interruption handler."""

import asyncio

import pytest

from src.audio.interruption_handler import InterruptionHandler, InterruptionType
from src.audio.vad_engine import SpeechState, VADResult
from src.config import VADConfig


class TestInterruptionHandler:
    """Tests for interruption detection and classification."""

    def setup_method(self):
        self.handler = InterruptionHandler(VADConfig())

    @pytest.mark.asyncio
    async def test_no_interruption_when_not_speaking(self):
        """No interruption if assistant is not speaking."""
        result = VADResult(
            is_speech=True,
            energy_db=-10.0,
            state=SpeechState.SPEAKING,
            speech_duration_ms=500,
        )
        interruption = await self.handler.on_vad_result(result)
        assert interruption == InterruptionType.NONE

    @pytest.mark.asyncio
    async def test_barge_in_detection(self):
        """Should detect barge-in when user speaks while assistant is speaking."""
        self.handler.set_assistant_speaking(True)

        result = VADResult(
            is_speech=True,
            energy_db=-15.0,
            state=SpeechState.SPEAKING,
            speech_duration_ms=800,  # Long enough for barge-in
        )
        interruption = await self.handler.on_vad_result(result)
        assert interruption == InterruptionType.BARGE_IN
        assert self.handler.should_cancel_tts

    @pytest.mark.asyncio
    async def test_noise_ignored(self):
        """Short speech bursts should be classified as noise."""
        self.handler.set_assistant_speaking(True)

        result = VADResult(
            is_speech=True,
            energy_db=-30.0,
            state=SpeechState.SPEAKING,
            speech_duration_ms=100,  # Too short
        )
        interruption = await self.handler.on_vad_result(result)
        assert interruption == InterruptionType.NOISE

    @pytest.mark.asyncio
    async def test_backchannel_detection(self):
        """Short, quiet speech should be classified as backchannel."""
        self.handler.set_assistant_speaking(True)

        result = VADResult(
            is_speech=True,
            energy_db=-35.0,  # Quiet
            state=SpeechState.SPEAKING,
            speech_duration_ms=400,  # Short
        )
        interruption = await self.handler.on_vad_result(result)
        assert interruption == InterruptionType.BACKCHANNEL

    @pytest.mark.asyncio
    async def test_tts_cancel_event(self):
        """TTS cancel event should be set on barge-in."""
        self.handler.set_assistant_speaking(True)
        assert not self.handler.should_cancel_tts

        result = VADResult(
            is_speech=True,
            energy_db=-10.0,
            state=SpeechState.SPEAKING,
            speech_duration_ms=800,
        )
        await self.handler.on_vad_result(result)
        assert self.handler.should_cancel_tts

    def test_reset_clears_state(self):
        """Reset should clear all interruption state."""
        self.handler.set_assistant_speaking(True)
        self.handler.reset()
        assert not self.handler.should_cancel_tts

    @pytest.mark.asyncio
    async def test_listener_notification(self):
        """Registered listeners should be notified of barge-in."""
        queue = self.handler.register_listener()
        self.handler.set_assistant_speaking(True)

        result = VADResult(
            is_speech=True,
            energy_db=-10.0,
            state=SpeechState.SPEAKING,
            speech_duration_ms=800,
        )
        await self.handler.on_vad_result(result)

        assert not queue.empty()
        notification = await queue.get()
        assert notification == InterruptionType.BARGE_IN
