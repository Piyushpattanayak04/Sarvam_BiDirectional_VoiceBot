"""Tests for the dialogue manager and conversation state."""

import pytest

from src.conversation.dialogue_manager import DialogueAction, DialogueManager
from src.conversation.state import ConversationContext, DialogueState
from src.nlu.entity_extractor import ExtractedEntities
from src.nlu.intent_detector import BankingIntent, IntentResult
from src.nlu.pipeline import NLUResult


def make_nlu_result(
    intent: BankingIntent,
    confidence: float = 0.9,
    days: int | None = None,
    card_type: str | None = None,
    loan_type: str | None = None,
) -> NLUResult:
    """Helper to create NLUResult for testing."""
    return NLUResult(
        intent=IntentResult(
            intent=intent,
            confidence=confidence,
            raw_text="test",
        ),
        entities=ExtractedEntities(
            days=days,
            card_type=card_type,
            loan_type=loan_type,
        ),
    )


class TestDialogueManager:
    """Tests for the dialogue state machine."""

    def setup_method(self):
        self.manager = DialogueManager()

    @pytest.mark.asyncio
    async def test_greeting_flow(self):
        """Greeting intent should return greeting response."""
        ctx = ConversationContext(session_id="test")
        ctx.transition_to(DialogueState.LISTENING)

        nlu = make_nlu_result(BankingIntent.GREETING)
        action = await self.manager.process_turn(nlu, ctx)

        assert "welcome" in action.response_text.lower() or "help" in action.response_text.lower()
        assert action.next_state == DialogueState.LISTENING

    @pytest.mark.asyncio
    async def test_block_card_with_card_type(self):
        """Block card with card type should ask for confirmation."""
        ctx = ConversationContext(session_id="test")
        ctx.transition_to(DialogueState.LISTENING)

        nlu = make_nlu_result(BankingIntent.BLOCK_CARD, card_type="debit")
        action = await self.manager.process_turn(nlu, ctx)

        assert "debit" in action.response_text.lower()
        assert "confirm" in action.response_text.lower() or "proceed" in action.response_text.lower()

    @pytest.mark.asyncio
    async def test_bank_statement_without_days(self):
        """Bank statement without days should ask for duration."""
        ctx = ConversationContext(session_id="test")
        ctx.transition_to(DialogueState.LISTENING)

        nlu = make_nlu_result(BankingIntent.BANK_STATEMENT)
        action = await self.manager.process_turn(nlu, ctx)

        assert "days" in action.response_text.lower() or "how many" in action.response_text.lower()
        assert not action.should_execute_api

    @pytest.mark.asyncio
    async def test_bank_statement_with_days(self):
        """Bank statement with days should trigger API execution."""
        ctx = ConversationContext(session_id="test")
        ctx.transition_to(DialogueState.LISTENING)

        nlu = make_nlu_result(BankingIntent.BANK_STATEMENT, days=30)
        action = await self.manager.process_turn(nlu, ctx)

        assert action.should_execute_api
        assert action.api_action == "bank_statement"
        assert action.api_params["days"] == 30

    @pytest.mark.asyncio
    async def test_loan_enquiry(self):
        """Loan enquiry should return loan information."""
        ctx = ConversationContext(session_id="test")
        ctx.transition_to(DialogueState.LISTENING)

        nlu = make_nlu_result(BankingIntent.LOAN_ENQUIRY, loan_type="personal")
        action = await self.manager.process_turn(nlu, ctx)

        assert "personal" in action.response_text.lower()
        assert "interest" in action.response_text.lower() or "loan" in action.response_text.lower()

    @pytest.mark.asyncio
    async def test_confirmation_executes_pending_action(self):
        """Confirmation should execute the pending block card."""
        ctx = ConversationContext(session_id="test")
        ctx.state = DialogueState.AWAITING_CONFIRMATION
        ctx.current_intent = BankingIntent.BLOCK_CARD.value
        ctx.current_entities = {"card_type": "debit"}

        nlu = make_nlu_result(BankingIntent.CONFIRMATION)
        action = await self.manager.process_turn(nlu, ctx)

        assert action.should_execute_api
        assert action.api_action == "block_card"

    @pytest.mark.asyncio
    async def test_denial_cancels_action(self):
        """Denial should cancel and ask what else to help with."""
        ctx = ConversationContext(session_id="test")
        ctx.state = DialogueState.AWAITING_CONFIRMATION
        ctx.current_intent = BankingIntent.BLOCK_CARD.value

        nlu = make_nlu_result(BankingIntent.DENIAL)
        action = await self.manager.process_turn(nlu, ctx)

        assert not action.should_execute_api
        assert "anything else" in action.response_text.lower() or "else" in action.response_text.lower()


class TestConversationContext:
    """Tests for conversation state management."""

    def test_valid_transitions(self):
        ctx = ConversationContext(session_id="test")
        assert ctx.state == DialogueState.IDLE
        assert ctx.transition_to(DialogueState.GREETING)
        assert ctx.state == DialogueState.GREETING

    def test_invalid_transitions(self):
        ctx = ConversationContext(session_id="test")
        assert not ctx.transition_to(DialogueState.EXECUTING)  # Can't go from IDLE to EXECUTING
        assert ctx.state == DialogueState.IDLE  # State unchanged

    def test_add_turn(self):
        ctx = ConversationContext(session_id="test")
        ctx.add_turn("user", "hello")
        ctx.add_turn("assistant", "hi there")
        assert len(ctx.turns) == 2
        assert ctx.turn_count == 2

    def test_context_summary(self):
        ctx = ConversationContext(session_id="test")
        ctx.add_turn("user", "hello")
        ctx.add_turn("assistant", "hi")
        summary = ctx.get_context_summary()
        assert "user: hello" in summary
        assert "assistant: hi" in summary

    def test_merge_entities(self):
        ctx = ConversationContext(session_id="test")
        ctx.merge_entities({"days": 30, "card_type": None})
        assert ctx.current_entities["days"] == 30
        assert "card_type" not in ctx.current_entities  # None values not merged
