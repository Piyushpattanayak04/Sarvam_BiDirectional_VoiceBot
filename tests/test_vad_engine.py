"""Tests for the VAD engine."""

import pytest

from src.audio.vad_engine import SpeechState, VADEngine


class TestVADEngine:
    """Unit tests for Voice Activity Detection."""

    def test_initial_state_is_silence(self, vad_config, audio_config, silence_frame):
        """VAD should start in silence state."""
        vad = VADEngine(vad_config, audio_config)
        result = vad.process_frame(silence_frame)
        assert result.state == SpeechState.SILENCE
        assert not result.is_speech
        assert not vad.is_speaking

    def test_speech_detection(self, vad_config, audio_config, speech_frame):
        """VAD should detect speech after enough consecutive speech frames."""
        vad = VADEngine(vad_config, audio_config)

        # Feed speech frames — need SPEECH_ONSET_FRAMES to trigger
        results = []
        for _ in range(10):
            result = vad.process_frame(speech_frame)
            results.append(result)

        # Should have transitioned to speaking at some point
        states = [r.state for r in results]
        assert SpeechState.SPEECH_START in states or SpeechState.SPEAKING in states

    def test_silence_after_speech(self, vad_config, audio_config, speech_frame, silence_frame):
        """VAD should detect end of speech after silence threshold."""
        vad = VADEngine(vad_config, audio_config)

        # Start speaking
        for _ in range(15):
            vad.process_frame(speech_frame)

        # Now silence
        results = []
        for _ in range(100):  # Well past the silence threshold
            result = vad.process_frame(silence_frame)
            results.append(result)

        states = [r.state for r in results]
        # Should eventually reach SPEECH_END
        has_end = SpeechState.SPEECH_END in states
        has_pause = (
            SpeechState.PAUSE_SHORT in states
            or SpeechState.PAUSE_MEDIUM in states
            or SpeechState.PAUSE_LONG in states
        )
        assert has_end or has_pause

    def test_energy_computation(self, vad_config, audio_config):
        """Energy computation should produce reasonable values."""
        vad = VADEngine(vad_config, audio_config)

        # Silence should have very low energy
        silence = b"\x00\x00" * 320
        energy = vad._compute_energy_db(silence)
        assert energy <= -90.0

        # Full-scale should be near 0 dB
        import struct
        loud = struct.pack("<h", 32000) * 320
        energy = vad._compute_energy_db(loud)
        assert energy > -5.0

    def test_reset_clears_state(self, vad_config, audio_config, speech_frame):
        """Reset should clear all VAD state."""
        vad = VADEngine(vad_config, audio_config)

        for _ in range(15):
            vad.process_frame(speech_frame)

        vad.reset()
        assert not vad.is_speaking
