"""
NLU Pipeline — Orchestrates intent detection and entity extraction.

Combines rule-based and LLM-based approaches into a unified pipeline
with configurable confidence fallback thresholds.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from groq import Groq

from src.config import AppConfig
from src.logging_config import get_logger
from src.nlu.entity_extractor import EntityExtractor, ExtractedEntities
from src.nlu.intent_detector import (
    BankingIntent,
    IntentResult,
    RuleBasedIntentDetector,
    build_llm_intent_prompt,
    parse_llm_intent_response,
)

logger = get_logger(__name__)

# If rule-based confidence is below this, escalate to LLM
RULE_CONFIDENCE_THRESHOLD = 0.3


@dataclass
class NLUResult:
    """Combined result from the NLU pipeline."""
    intent: IntentResult
    entities: ExtractedEntities
    requires_clarification: bool = False
    clarification_prompt: str = ""


class NLUPipeline:
    """
    Unified NLU pipeline:
        Text → Rule-based intent (fast path)
             → If low confidence → LLM intent classification
             → Entity extraction (always runs)
             → Merge results → NLUResult
    """

    def __init__(self, app_config: AppConfig) -> None:
        self._rule_detector = RuleBasedIntentDetector()
        self._entity_extractor = EntityExtractor()
        self._app_config = app_config
        self._client: Groq | None = None
        # Cache last 50 LLM results to avoid redundant API calls for identical utterances
        self._llm_cache: dict[str, IntentResult] = {}
        self._cache_max_size = 50

    def _get_client(self) -> Groq:
        if self._client is None:
            self._client = Groq(api_key=self._app_config.groq.api_key)
        return self._client

    async def process(
        self,
        text: str,
        conversation_context: str = "",
    ) -> NLUResult:
        """
        Full NLU processing pipeline.
        
        Args:
            text: Transcribed user utterance
            conversation_context: Previous conversation turns for disambiguation
            
        Returns:
            NLUResult with intent, entities, and clarification needs
        """
        # Step 1: Rule-based intent detection (< 1ms)
        rule_result = self._rule_detector.detect(text)
        logger.debug(
            "rule_intent_result",
            intent=rule_result.intent.value,
            confidence=rule_result.confidence,
        )

        # Step 2: Entity extraction (< 1ms)
        entities = self._entity_extractor.extract(text)

        # Step 3: Decide if LLM fallback is needed
        intent_result = rule_result
        if rule_result.confidence < RULE_CONFIDENCE_THRESHOLD:
            if self._is_likely_hallucination(text):
                logger.info("hallucination_detected_skipping_llm", text=text[:60])
            else:
                cache_key = text.strip().lower()
                if cache_key in self._llm_cache:
                    logger.debug("llm_cache_hit", key=cache_key[:40])
                    intent_result = self._llm_cache[cache_key]
                else:
                    logger.info(
                        "escalating_to_llm",
                        rule_intent=rule_result.intent.value,
                        rule_confidence=rule_result.confidence,
                    )
                    try:
                        llm_result = await self._llm_classify(text, conversation_context)
                        if llm_result.confidence > rule_result.confidence:
                            intent_result = llm_result
                            # Merge LLM entities if they provide additional info
                            if "days" in llm_result.entities and entities.days is None:
                                entities.days = int(llm_result.entities["days"])
                            if "loan_type" in llm_result.entities and entities.loan_type is None:
                                entities.loan_type = str(llm_result.entities["loan_type"])
                        # Store result in cache (evict oldest entry if full)
                        if len(self._llm_cache) >= self._cache_max_size:
                            self._llm_cache.pop(next(iter(self._llm_cache)))
                        self._llm_cache[cache_key] = intent_result
                    except Exception as e:
                        logger.error("llm_classification_failed", error=str(e))
                        # Fall back to rule result

        # Step 4: Check if clarification is needed
        requires_clarification, clarification_prompt = self._check_clarification(
            intent_result, entities
        )

        return NLUResult(
            intent=intent_result,
            entities=entities,
            requires_clarification=requires_clarification,
            clarification_prompt=clarification_prompt,
        )

    async def _llm_classify(
        self, text: str, conversation_context: str
    ) -> IntentResult:
        """Use Groq llama3-8b-8192 for fast, low-latency intent classification."""
        client = self._get_client()
        messages = build_llm_intent_prompt(text, conversation_context)

        # Run the synchronous SDK call in a thread pool so it doesn't block
        # the event loop. GPT-4o-mini with max_tokens=100 typically responds in <1s.
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model=self._app_config.groq.llm_model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=100,
                    temperature=0,
                ),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            logger.warning("llm_classify_timeout", timeout_s=3.0)
            raise

        response_text = response.choices[0].message.content  # type: ignore[union-attr]
        return parse_llm_intent_response(response_text or "", text)

    @staticmethod
    def _is_likely_hallucination(text: str) -> bool:
        """Detect repetitive STT hallucinations (e.g. 'हम लोग हैं कि हम लोग हैं...')."""
        words = text.split()
        if len(words) < 8:
            return False
        return len(set(words)) / len(words) < 0.3

    def _check_clarification(
        self,
        intent: IntentResult,
        entities: ExtractedEntities,
    ) -> tuple[bool, str]:
        """
        Determine if we need to ask the user for more information.

        Returns (needs_clarification, prompt_text).
        """
        if intent.intent == BankingIntent.UNKNOWN:
            return True, (
                "I'm sorry, I didn't quite understand. Could you please tell me "
                "if you'd like to block a card, get a bank statement, or enquire about a loan?"
            )

        if intent.intent == BankingIntent.BANK_STATEMENT and entities.days is None:
            return True, "For how many days would you like the bank statement?"

        if intent.intent == BankingIntent.BLOCK_CARD and entities.card_type is None:
            return True, "Would you like to block your debit card or credit card?"

        return False, ""
