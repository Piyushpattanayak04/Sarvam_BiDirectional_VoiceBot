"""
Voice Session Orchestrator

The master controller for a single voice conversation session.
Ties together all subsystems:
  - Audio stream manager (VAD, interruptions, buffering)
  - Sarvam STT (speech-to-text streaming)
  - NLU pipeline (intent + entity extraction)
  - Dialogue manager (conversation state machine)
  - Banking API client (backend execution)
  - Sarvam TTS (text-to-speech streaming)

This is the "session loop" — one instance per concurrent user.

Pipeline Architecture (per session):
    ┌──────────────────────────────────────────────────────────┐
    │                   WebSocket Connection                    │
    │  ┌─────────┐                            ┌─────────────┐  │
    │  │ Mic In  │──→ PCM Frames ──→ VAD ──→ │ STT Queue   │  │
    │  └─────────┘                            └──────┬──────┘  │
    │                                                │         │
    │                                    ┌───────────▼───────┐ │
    │                                    │   Sarvam STT      │ │
    │                                    │   (Streaming)     │ │
    │                                    └───────────┬───────┘ │
    │                                                │         │
    │                                    ┌───────────▼───────┐ │
    │                                    │   NLU Pipeline    │ │
    │                                    │   (Intent+Entity) │ │
    │                                    └───────────┬───────┘ │
    │                                                │         │
    │                                    ┌───────────▼───────┐ │
    │                                    │ Dialogue Manager  │ │
    │                                    │ (State Machine)   │ │
    │                                    └───────────┬───────┘ │
    │                                                │         │
    │                                    ┌───────────▼───────┐ │
    │                                    │  Banking API      │ │
    │                                    │  (if needed)      │ │
    │                                    └───────────┬───────┘ │
    │                                                │         │
    │                                    ┌───────────▼───────┐ │
    │  ┌─────────┐                       │   Sarvam TTS     │ │
    │  │ Speaker │←── TTS Audio ←────────│   (Streaming)    │ │
    │  └─────────┘                       └──────────────────┘ │
    └──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import re
import time

from sarvamai import AsyncSarvamAI, SarvamAI

from src.audio.stream_manager import AudioStreamManager
from src.banking.api_client import (
    BankingService,
    BankStatementRequest,
    BlockCardRequest,
    LoanEnquiryRequest,
)
from src.config import AppConfig
from src.conversation.dialogue_manager import DialogueManager
from src.conversation.session_store import InMemorySessionStore, SessionStore
from src.conversation.state import ConversationContext, DialogueState
from src.logging_config import get_logger
from src.nlu.pipeline import NLUPipeline

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Language detection → TTS language code mapping
# ──────────────────────────────────────────────────────────────────────

LANGUAGE_MAP = {
    "en": "en-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "gu": "gu-IN",
    "hi": "hi-IN",
    "bn": "bn-IN",
    "or": "or-IN",
    "mixed": "en-IN",  # Default to English for mixed
}

# Unicode script ranges for language detection fallback
_SCRIPT_RANGES = [
    (re.compile(r"[\u0C00-\u0C7F]"), "te"),   # Telugu
    (re.compile(r"[\u0B80-\u0BFF]"), "ta"),   # Tamil
    (re.compile(r"[\u0A80-\u0AFF]"), "gu"),   # Gujarati
    (re.compile(r"[\u0900-\u097F]"), "hi"),   # Devanagari (Hindi)
    (re.compile(r"[\u0980-\u09FF]"), "bn"),   # Bengali
    (re.compile(r"[\u0B00-\u0B7F]"), "or"),   # Odia
]


def _detect_language_from_script(text: str) -> str | None:
    """Return a 2-letter language code if the text contains a recognisable non-Latin script."""
    for pattern, lang in _SCRIPT_RANGES:
        if pattern.search(text):
            return lang
    return None


class VoiceSession:
    """
    Manages a single bidirectional voice conversation.
    
    Lifecycle:
      1. Client connects via WebSocket
      2. VoiceSession is created and started
      3. Two async tasks run:
         a. Input loop: receives audio → VAD → STT → NLU → Dialogue → response
         b. Output loop: streams TTS audio back to client
      4. On disconnect, session is persisted for potential resumption
    """

    def __init__(
        self,
        session_id: str,
        config: AppConfig,
        session_store: SessionStore | None = None,
    ) -> None:
        self._session_id = session_id
        self._config = config
        self._session_store = session_store or InMemorySessionStore()

        # Subsystems
        self._audio_manager = AudioStreamManager(config.audio, config.vad)
        self._nlu = NLUPipeline(config)
        self._dialogue = DialogueManager()
        self._banking = BankingService()

        # Sarvam clients for STT/TTS
        self._sarvam: SarvamAI | None = None
        self._async_sarvam: AsyncSarvamAI | None = None

        # Session context
        self._context: ConversationContext | None = None

        # Control
        self._running = False
        self._tasks: list[asyncio.Task] = []  # type: ignore[type-arg]

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def audio_manager(self) -> AudioStreamManager:
        return self._audio_manager

    def _get_sarvam(self) -> SarvamAI:
        if self._sarvam is None:
            self._sarvam = SarvamAI(
                api_subscription_key=self._config.sarvam.api_key
            )
        return self._sarvam

    def _get_async_sarvam(self) -> AsyncSarvamAI:
        if self._async_sarvam is None:
            self._async_sarvam = AsyncSarvamAI(
                api_subscription_key=self._config.sarvam.api_key
            )
        return self._async_sarvam

    async def start(self) -> None:
        """Initialize and start the voice session."""
        self._running = True

        # Load or create conversation context
        existing = await self._session_store.get(self._session_id)
        if existing:
            self._context = existing
            logger.info("session_resumed", session_id=self._session_id)
        else:
            self._context = ConversationContext(
                session_id=self._session_id,
                created_at_ms=time.time() * 1000,
            )
            logger.info("session_created", session_id=self._session_id)

        await self._audio_manager.start()

        # Start the STT processing task
        stt_task = asyncio.create_task(self._stt_processing_loop())
        self._tasks.append(stt_task)

        # Send greeting
        await self._send_greeting()

    async def stop(self) -> None:
        """Stop the session and persist state."""
        self._running = False
        await self._audio_manager.stop()

        for task in self._tasks:
            task.cancel()

        if self._context:
            await self._session_store.save(self._context)
            logger.info(
                "session_stopped",
                session_id=self._session_id,
                turns=self._context.turn_count,
            )

    async def on_audio_frame(self, pcm_bytes: bytes) -> None:
        """Handle an incoming audio frame from the client."""
        if not self._running:
            return
        await self._audio_manager.process_input_frame(pcm_bytes)

    async def _send_greeting(self) -> None:
        """Send the initial greeting."""
        if self._context is None:
            return

        self._context.transition_to(DialogueState.GREETING)
        lang = self._context.preferred_language
        greeting = self._dialogue.get_response_template("greeting", lang)

        await self._synthesize_and_stream(greeting)
        self._context.add_turn("assistant", greeting)
        self._context.transition_to(DialogueState.LISTENING)

    async def _stt_processing_loop(self) -> None:
        """
        Main processing loop: consumes speech segments from VAD,
        runs STT + NLU + Dialogue, and generates responses.
        """
        while self._running:
            try:
                # Wait for a speech segment from VAD
                segment = await asyncio.wait_for(
                    self._audio_manager.stt_queue.get(),
                    timeout=30.0,
                )

                if not segment.is_final:
                    # Partial segment — could start preliminary processing
                    # For now, we wait for the final segment
                    continue

                # ── Step 1: Speech-to-Text ───────────────────────────
                start_time = time.monotonic()
                transcript = await self._transcribe(segment.pcm_data)

                if not transcript or not transcript.strip():
                    continue

                stt_latency = (time.monotonic() - start_time) * 1000
                logger.info(
                    "stt_complete",
                    text=transcript,
                    latency_ms=round(stt_latency),
                )

                if self._context:
                    self._context.add_turn("user", transcript)

                # ── Step 2: NLU Processing ───────────────────────────
                nlu_start = time.monotonic()
                context_summary = (
                    self._context.get_context_summary() if self._context else ""
                )
                nlu_result = await self._nlu.process(transcript, context_summary)
                nlu_latency = (time.monotonic() - nlu_start) * 1000

                logger.info(
                    "nlu_complete",
                    intent=nlu_result.intent.intent.value,
                    confidence=nlu_result.intent.confidence,
                    entities=nlu_result.entities.__dict__,
                    latency_ms=round(nlu_latency),
                )

                # Update language detection — prefer LLM result, fall back to script detection
                if self._context:
                    detected = nlu_result.intent.detected_language
                    if detected == "en" or detected == "mixed":
                        # LLM didn't confidently identify a non-English language;
                        # use Unicode script ranges as a reliable fallback
                        script_lang = _detect_language_from_script(transcript)
                        if script_lang:
                            detected = script_lang
                    if detected and detected != "en":
                        self._context.detected_language = detected
                        self._context.preferred_language = LANGUAGE_MAP.get(
                            detected, "en-IN"
                        )

                # ── Step 3: Dialogue Processing ──────────────────────
                if self._context:
                    self._context.transition_to(DialogueState.PROCESSING)
                    action = await self._dialogue.process_turn(
                        nlu_result, self._context
                    )

                    # ── Step 4: Execute Banking API if needed ────────
                    if action.should_execute_api:
                        await self._execute_banking_action(
                            action.api_action, action.api_params
                        )

                    # ── Step 5: Generate voice response ──────────────
                    if action.response_text:
                        await self._synthesize_and_stream(action.response_text)
                        self._context.add_turn("assistant", action.response_text)

                    # Update state
                    self._context.transition_to(action.next_state)
                    await self._session_store.save(self._context)

                    total_latency = (time.monotonic() - start_time) * 1000
                    logger.info(
                        "turn_complete",
                        total_latency_ms=round(total_latency),
                        target_ms=self._config.response_timeout_ms,
                    )

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("stt_loop_error", error=str(e), exc_info=True)
                await asyncio.sleep(0.5)

    async def _transcribe(self, pcm_data: bytes) -> str:
        """
        Transcribe PCM audio using Sarvam Saaras STT.
        
        Uses the batch (non-streaming) API for complete utterances.
        In production with LiveKit, this would use the streaming STT plugin.
        """
        try:
            client = self._get_sarvam()

            # Determine language hint
            lang = "unknown"  # Auto-detect
            if self._context and self._context.detected_language != "en":
                lang_map = {
                    "ta": "ta-IN",
                    "te": "te-IN",
                    "gu": "gu-IN",
                    "hi": "hi-IN",
                    "bn": "bn-IN",
                    "or": "or-IN",
                }
                lang = lang_map.get(self._context.detected_language, "unknown")

            # Save PCM to temp WAV file for the batch API
            import io
            import tempfile
            import wave

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                with wave.open(f.name, "wb") as wf:
                    wf.setnchannels(self._config.audio.channels)
                    wf.setsampwidth(self._config.audio.bit_depth // 8)
                    wf.setframerate(self._config.audio.sample_rate)
                    wf.writeframes(pcm_data)

            # Read file into memory, then delete temp file before calling API
            with open(temp_path, "rb") as fh:
                wav_bytes = fh.read()

            import os
            os.unlink(temp_path)

            # Run sync SDK call in a thread to avoid blocking the event loop
            response = await asyncio.to_thread(
                client.speech_to_text.transcribe,
                file=io.BytesIO(wav_bytes),
                model=self._config.sarvam.stt_model,
                language_code=lang,
            )

            return response.transcript or ""

        except Exception as e:
            logger.error("stt_error", error=str(e))
            return ""

    async def _synthesize_and_stream(self, text: str) -> None:
        """
        Convert text to speech using Sarvam Bulbul TTS and stream to client.

        The synchronous batch SDK call runs in a thread pool (asyncio.to_thread)
        so it does not block the event loop. A race against the barge-in cancel
        event lets us abort early if the user interrupts.

        Note: the streaming WebSocket TTS API sends raw partial MP3 frames that
        browsers cannot independently decode with decodeAudioData. The batch API
        returns complete WAV blobs that work correctly on the client.
        """
        import base64

        interruption_handler = self._audio_manager.interruption_handler

        # Skip entirely if the user already barged in before we started
        if interruption_handler.should_cancel_tts:
            logger.info("tts_skipped_interruption")
            return

        try:
            interruption_handler.set_assistant_speaking(True)
            client = self._get_sarvam()

            lang = "en-IN"
            if self._context:
                lang = self._context.preferred_language

            # Race the (synchronous) TTS API call against the interruption event.
            # asyncio.to_thread runs the blocking call in a thread pool; wrapping
            # it in create_task lets us cancel it (the thread keeps running briefly,
            # but we stop waiting for it and discard the result).
            tts_task = asyncio.create_task(
                asyncio.to_thread(
                    client.text_to_speech.convert,
                    text=text,
                    target_language_code=lang,
                    model=self._config.sarvam.tts_model,
                    speaker=self._config.sarvam.tts_default_speaker,
                )
            )
            cancel_event = interruption_handler.tts_cancel_event
            cancel_wait = asyncio.create_task(cancel_event.wait())

            done, pending = await asyncio.wait(
                {tts_task, cancel_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Clean up the task we won't use
            for task in pending:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

            if cancel_wait in done:
                # Barge-in fired while waiting for TTS — stop immediately
                logger.info("tts_cancelled_by_interruption")
                await self._audio_manager.signal_tts_end()
                return

            # TTS completed — stream audio chunks to client
            response = tts_task.result()
            if hasattr(response, "audios") and response.audios:
                for audio_b64 in response.audios:
                    if interruption_handler.should_cancel_tts:
                        logger.info("tts_cancelled_by_interruption")
                        break
                    audio_bytes = base64.b64decode(audio_b64)
                    await self._audio_manager.enqueue_tts_chunk(audio_bytes)

            await self._audio_manager.signal_tts_end()

        except Exception as e:
            logger.error("tts_error", error=str(e))
            interruption_handler.set_assistant_speaking(False)

    async def _execute_banking_action(
        self, action: str, params: dict
    ) -> None:
        """Execute a banking API action."""
        try:
            match action:
                case "block_card":
                    request = BlockCardRequest(
                        user_id=params.get("user_id", "user_001"),
                        card_type=params.get("card_type", "debit"),
                    )
                    result = await self._banking.block_card(request)

                case "bank_statement":
                    request = BankStatementRequest(
                        user_id=params.get("user_id", "user_001"),
                        days=params.get("days", 30),
                    )
                    result = await self._banking.get_bank_statement(request)

                case "loan_enquiry":
                    request = LoanEnquiryRequest(
                        user_id=params.get("user_id", "user_001"),
                        loan_type=params.get("loan_type", "personal"),
                    )
                    result = await self._banking.loan_enquiry(request)

                case _:
                    logger.warning("unknown_banking_action", action=action)
                    return

            logger.info(
                "banking_api_result",
                action=action,
                status=result.status.value,
                latency_ms=round(result.latency_ms),
            )

        except Exception as e:
            logger.error("banking_api_error", action=action, error=str(e))
