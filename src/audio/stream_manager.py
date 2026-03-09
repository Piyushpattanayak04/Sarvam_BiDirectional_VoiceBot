"""
Audio Stream Manager

Manages the bidirectional audio stream between client and server.
Responsible for:
  - Receiving raw PCM audio from client WebSocket
  - Feeding frames to VAD engine
  - Accumulating speech segments for STT
  - Streaming TTS audio back to client
  - Coordinating interruptions

This is the central orchestrator for the audio pipeline.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np

from src.audio.interruption_handler import InterruptionHandler, InterruptionType
from src.audio.vad_engine import SpeechState, VADEngine
from src.config import AudioConfig, VADConfig
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AudioSegment:
    """A captured speech segment ready for STT processing."""
    pcm_data: bytes
    duration_ms: float
    is_final: bool = False  # True if this is a complete utterance


@dataclass
class StreamMetrics:
    """Real-time metrics for the audio stream."""
    frames_processed: int = 0
    speech_segments_emitted: int = 0
    interruptions_detected: int = 0
    total_speech_ms: float = 0.0
    total_silence_ms: float = 0.0


class AudioStreamManager:
    """
    Bidirectional audio stream orchestrator.

    Data Flow:
        Client WebSocket
          → Raw PCM chunks (20ms frames)
          → VAD Engine (speech/silence classification)
          → Speech Accumulator (collect frames during speech)
          → STT Queue (complete utterances for transcription)

        TTS Response
          → TTS audio chunks
          → Interruption check per chunk
          → Client WebSocket

    Key Design Decisions:
        1. Audio is processed frame-by-frame (20ms) for minimal latency
        2. Speech segments are accumulated in a bytearray buffer
        3. Partial segments can be emitted on PAUSE_MEDIUM for early STT
        4. TTS output is streamed chunk-by-chunk with interruption checks
    """

    def __init__(
        self,
        audio_config: AudioConfig,
        vad_config: VADConfig,
    ) -> None:
        self._audio_config = audio_config
        self._vad = VADEngine(vad_config, audio_config)
        self._interruption_handler = InterruptionHandler(vad_config)
        
        # Speech accumulation buffer
        self._speech_buffer = bytearray()
        self._speech_start_ms: float = 0.0
        self._partial_emitted = False  # Track if we already emitted a partial
        
        # Output queues
        self._stt_queue: asyncio.Queue[AudioSegment] = asyncio.Queue(maxsize=50)
        self._tts_output_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=200)
        
        self._metrics = StreamMetrics()
        self._running = False

    @property
    def stt_queue(self) -> asyncio.Queue[AudioSegment]:
        """Queue of speech segments for STT processing."""
        return self._stt_queue

    @property
    def tts_output_queue(self) -> asyncio.Queue[bytes | None]:
        """Queue of TTS audio chunks to send to client. None = end of response."""
        return self._tts_output_queue

    @property
    def interruption_handler(self) -> InterruptionHandler:
        return self._interruption_handler

    @property
    def metrics(self) -> StreamMetrics:
        return self._metrics

    async def start(self) -> None:
        """Start the audio stream manager."""
        self._running = True
        self._vad.reset()
        self._interruption_handler.reset()
        logger.info("audio_stream_started")

    async def stop(self) -> None:
        """Stop the audio stream manager and flush buffers."""
        self._running = False
        # Flush any remaining speech
        if len(self._speech_buffer) > 0:
            await self._emit_speech_segment(is_final=True)
        # Signal end of TTS
        await self._tts_output_queue.put(None)
        logger.info("audio_stream_stopped", metrics=self._metrics.__dict__)

    async def process_input_frame(self, pcm_bytes: bytes) -> None:
        """
        Process a single incoming audio frame from the client.
        
        This is the hot path — must complete in <1ms per frame.
        """
        if not self._running:
            return

        self._metrics.frames_processed += 1

        # Run VAD
        vad_result = self._vad.process_frame(pcm_bytes)

        # Check for interruptions
        interruption = await self._interruption_handler.on_vad_result(vad_result)
        if interruption == InterruptionType.BARGE_IN:
            self._metrics.interruptions_detected += 1
            # Clear TTS output queue so no more audio is sent to client
            while not self._tts_output_queue.empty():
                try:
                    self._tts_output_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            # Immediately inject the end-of-stream sentinel so _output_loop
            # tells the client to stop playback right now, without waiting
            # for _synthesize_and_stream to notice the cancellation.
            try:
                self._tts_output_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        # Accumulate speech or emit segments based on state
        match vad_result.state:
            case SpeechState.SPEECH_START:
                self._speech_buffer.clear()
                self._speech_start_ms = vad_result.speech_duration_ms
                self._speech_buffer.extend(pcm_bytes)
                self._partial_emitted = False

            case SpeechState.SPEAKING:
                self._speech_buffer.extend(pcm_bytes)

            case SpeechState.PAUSE_SHORT:
                # Still accumulating — user might continue
                self._speech_buffer.extend(pcm_bytes)

            case SpeechState.PAUSE_MEDIUM:
                # Emit partial segment once for early STT processing
                self._speech_buffer.extend(pcm_bytes)
                if len(self._speech_buffer) > 0 and not self._partial_emitted:
                    await self._emit_speech_segment(is_final=False)
                    self._partial_emitted = True

            case SpeechState.SPEECH_END | SpeechState.PAUSE_LONG:
                # Definite end — emit final segment
                if len(self._speech_buffer) > 0:
                    await self._emit_speech_segment(is_final=True)
                self._metrics.total_silence_ms += vad_result.silence_duration_ms

            case SpeechState.SILENCE:
                self._metrics.total_silence_ms += self._audio_config.chunk_ms

    async def enqueue_tts_chunk(self, audio_chunk: bytes) -> bool:
        """
        Enqueue a TTS audio chunk for playback to the client.
        
        Returns False if TTS was cancelled (interruption occurred).
        """
        if self._interruption_handler.should_cancel_tts:
            logger.debug("tts_chunk_dropped_interruption")
            return False

        try:
            self._tts_output_queue.put_nowait(audio_chunk)
            return True
        except asyncio.QueueFull:
            logger.warning("tts_output_queue_full")
            return False

    async def signal_tts_end(self) -> None:
        """Signal end of current TTS response.

        Puts a None sentinel into the output queue.  The websocket output loop
        will call interruption_handler.set_assistant_speaking(False) *after*
        it has sent all audio to the client, so barge-in detection stays active
        during the full playback window rather than deactivating the moment the
        last chunk is enqueued on the server side.
        """
        await self._tts_output_queue.put(None)

    async def _emit_speech_segment(self, is_final: bool) -> None:
        """Package accumulated speech and push to STT queue."""
        pcm_data = bytes(self._speech_buffer)
        duration_ms = len(pcm_data) / (
            self._audio_config.sample_rate
            * self._audio_config.channels
            * (self._audio_config.bit_depth // 8)
        ) * 1000

        segment = AudioSegment(
            pcm_data=pcm_data,
            duration_ms=duration_ms,
            is_final=is_final,
        )

        try:
            self._stt_queue.put_nowait(segment)
            self._metrics.speech_segments_emitted += 1
            self._metrics.total_speech_ms += duration_ms
            logger.debug(
                "speech_segment_emitted",
                duration_ms=round(duration_ms),
                is_final=is_final,
                buffer_size=len(pcm_data),
            )
        except asyncio.QueueFull:
            logger.warning("stt_queue_full_dropping_segment")

        if is_final:
            self._speech_buffer.clear()
