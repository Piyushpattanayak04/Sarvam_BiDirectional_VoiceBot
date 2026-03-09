"""Tests for intent detection and entity extraction."""

import pytest

from src.nlu.entity_extractor import EntityExtractor
from src.nlu.intent_detector import BankingIntent, RuleBasedIntentDetector


class TestRuleBasedIntentDetector:
    """Tests for the rule-based intent classification."""

    def setup_method(self):
        self.detector = RuleBasedIntentDetector()

    # ── Block Card ────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "text",
        [
            "I want to block my debit card",
            "Please freeze my credit card",
            "My card was stolen",
            "I lost my debit card yesterday",
            "Block my ATM card immediately",
            "I need to disable my card",
        ],
    )
    def test_block_card_english(self, text):
        result = self.detector.detect(text)
        assert result.intent == BankingIntent.BLOCK_CARD
        assert result.confidence > 0.5

    @pytest.mark.parametrize(
        "text",
        [
            "என் அட்டை தொலைந்தது",  # Tamil: My card is lost
            "కార్డ్ బ్లాక్ చేయండి",   # Telugu: Block the card
            "કાર્ડ બ્લોક કરો",        # Gujarati: Block the card
        ],
    )
    def test_block_card_multilingual(self, text):
        result = self.detector.detect(text)
        assert result.intent == BankingIntent.BLOCK_CARD

    # ── Bank Statement ────────────────────────────────────────────

    @pytest.mark.parametrize(
        "text",
        [
            "I need my bank statement",
            "Show me my account statement",
            "Get my transactions for last 30 days",
            "I want statement for past 15 days",
            "Give me my bank statement",
        ],
    )
    def test_bank_statement_english(self, text):
        result = self.detector.detect(text)
        assert result.intent == BankingIntent.BANK_STATEMENT

    @pytest.mark.parametrize(
        "text",
        [
            "எனக்கு கடந்த 10 நாட்களுக்கு வங்கி ஸ்டேட்மென்ட் வேண்டும்",  # Tamil
            "నాకు గత 15 రోజుల బ్యాంక్ స్టేట్మెంట్ కావాలి",               # Telugu
            "મને છેલ્લા 7 દિવસનું બેંક સ્ટેટમેન્ટ જોઈએ",                 # Gujarati
        ],
    )
    def test_bank_statement_multilingual(self, text):
        result = self.detector.detect(text)
        assert result.intent == BankingIntent.BANK_STATEMENT

    # ── Loan Enquiry ──────────────────────────────────────────────

    @pytest.mark.parametrize(
        "text",
        [
            "I want to know about personal loan",
            "What are your home loan interest rates",
            "Tell me about car loan options",
            "Loan enquiry",
            "I need information about education loan",
        ],
    )
    def test_loan_enquiry_english(self, text):
        result = self.detector.detect(text)
        assert result.intent == BankingIntent.LOAN_ENQUIRY

    # ── Greeting ──────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "text",
        [
            "Hello",
            "Hi",
            "Good morning",
            "Namaste",
        ],
    )
    def test_greeting(self, text):
        result = self.detector.detect(text)
        assert result.intent == BankingIntent.GREETING

    # ── Confirmation / Denial ─────────────────────────────────────

    @pytest.mark.parametrize("text", ["yes", "ok", "sure", "proceed", "ஆமாம்", "అవును", "હા"])
    def test_confirmation(self, text):
        result = self.detector.detect(text)
        assert result.intent == BankingIntent.CONFIRMATION

    @pytest.mark.parametrize("text", ["no", "cancel", "stop", "வேண்டாம்", "వద్దు", "ના"])
    def test_denial(self, text):
        result = self.detector.detect(text)
        assert result.intent == BankingIntent.DENIAL

    # ── Unknown ───────────────────────────────────────────────────

    def test_unknown_intent(self):
        result = self.detector.detect("What's the weather like today")
        assert result.intent == BankingIntent.UNKNOWN
        assert result.confidence == 0.0

    def test_empty_text(self):
        result = self.detector.detect("")
        assert result.intent == BankingIntent.UNKNOWN


class TestEntityExtractor:
    """Tests for dynamic entity extraction."""

    def setup_method(self):
        self.extractor = EntityExtractor()

    # ── Day extraction ────────────────────────────────────────────

    @pytest.mark.parametrize(
        "text,expected_days",
        [
            ("last 30 days", 30),
            ("past 15 days", 15),
            ("previous 7 days", 7),
            ("last week", 7),
            ("last month", 30),
            ("last year", 365),
            ("10 days ago", 10),
            ("last 2 weeks", 14),
            ("last 3 months", 90),
        ],
    )
    def test_english_day_extraction(self, text, expected_days):
        result = self.extractor.extract(text)
        assert result.days == expected_days

    def test_tamil_day_extraction(self):
        text = "கடந்த 10 நாட்களுக்கு"
        result = self.extractor.extract(text)
        assert result.days == 10

    def test_telugu_day_extraction(self):
        text = "గత 15 రోజుల"
        result = self.extractor.extract(text)
        assert result.days == 15

    def test_gujarati_day_extraction(self):
        text = "છેલ્લા 7 દિવસનું"
        result = self.extractor.extract(text)
        assert result.days == 7

    # ── Card type extraction ──────────────────────────────────────

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("block my debit card", "debit"),
            ("freeze credit card", "credit"),
            ("block atm card", "atm"),
        ],
    )
    def test_card_type_extraction(self, text, expected):
        result = self.extractor.extract(text)
        assert result.card_type == expected

    # ── Loan type extraction ──────────────────────────────────────

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("personal loan enquiry", "personal"),
            ("home loan interest rate", "home"),
            ("car loan details", "car"),
            ("education loan information", "education"),
            ("business loan options", "business"),
        ],
    )
    def test_loan_type_extraction(self, text, expected):
        result = self.extractor.extract(text)
        assert result.loan_type == expected

    def test_no_entities(self):
        result = self.extractor.extract("hello how are you")
        assert result.days is None
        assert result.card_type is None
        assert result.loan_type is None
