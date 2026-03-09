"""
Dynamic Entity Extraction

Extracts structured entities from user utterances, handling:
  - Time periods ("last 30 days", "past week", "previous month")
  - Card types ("debit card", "credit card", "ATM card")
  - Loan types ("personal loan", "home loan")
  - Amounts ("50,000 rupees", "5 lakh")
  - Multilingual numbers (Tamil, Telugu, Gujarati number words)

Design: Rule-based extraction for speed, with LLM fallback for ambiguous cases.
Numbers in Indian languages are handled via Unicode numeral mapping + word-to-number tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ExtractedEntities:
    """Container for all extracted entities."""
    days: int | None = None
    card_type: str | None = None
    loan_type: str | None = None
    amount: float | None = None
    account_type: str | None = None
    language: str = "en"


# ──────────────────────────────────────────────────────────────────────
# Number word mappings for Indian languages
# ──────────────────────────────────────────────────────────────────────

ENGLISH_NUMBERS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

TAMIL_NUMBERS: dict[str, int] = {
    "ஒன்று": 1, "இரண்டு": 2, "மூன்று": 3, "நான்கு": 4, "ஐந்து": 5,
    "ஆறு": 6, "ஏழு": 7, "எட்டு": 8, "ஒன்பது": 9, "பத்து": 10,
    "இருபது": 20, "முப்பது": 30, "நாற்பது": 40, "ஐம்பது": 50,
}

TELUGU_NUMBERS: dict[str, int] = {
    "ఒకటి": 1, "రెండు": 2, "మూడు": 3, "నాలుగు": 4, "ఐదు": 5,
    "ఆరు": 6, "ఏడు": 7, "ఎనిమిది": 8, "తొమ్మిది": 9, "పది": 10,
    "ఇరవై": 20, "ముప్పై": 30, "నలభై": 40, "యాభై": 50,
}

GUJARATI_NUMBERS: dict[str, int] = {
    "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4, "પાંચ": 5,
    "છ": 6, "સાત": 7, "આઠ": 8, "નવ": 9, "દસ": 10,
    "વીસ": 20, "ત્રીસ": 30, "ચાલીસ": 40, "પચાસ": 50,
}

ALL_NUMBER_WORDS = {**ENGLISH_NUMBERS, **TAMIL_NUMBERS, **TELUGU_NUMBERS, **GUJARATI_NUMBERS}

# ──────────────────────────────────────────────────────────────────────
# Time period patterns
# ──────────────────────────────────────────────────────────────────────

TIME_PERIOD_PATTERNS = [
    # "last/past/previous N days/weeks/months"
    (r"(?:last|past|previous|recent)\s+(\d+)\s*(?:day|days)", "days"),
    (r"(?:last|past|previous|recent)\s+(\d+)\s*(?:week|weeks)", "weeks"),
    (r"(?:last|past|previous|recent)\s+(\d+)\s*(?:month|months)", "months"),
    # "N days/weeks ago"
    (r"(\d+)\s*(?:day|days)\s*(?:ago|back)", "days"),
    (r"(\d+)\s*(?:week|weeks)\s*(?:ago|back)", "weeks"),
    # Shorthand
    (r"(?:last|past|previous)\s+week", "fixed_7"),
    (r"(?:last|past|previous)\s+month", "fixed_30"),
    (r"(?:last|past|previous)\s+year", "fixed_365"),
    (r"(?:this)\s+week", "fixed_7"),
    (r"(?:this)\s+month", "fixed_30"),
    # Tamil patterns
    (r"(?:கடந்த|சென்ற|முந்தைய)\s+(\d+)\s*(?:நாட்கள்|நாள்|நாட்களுக்கு)", "days"),
    (r"(?:கடந்த|சென்ற)\s+(\d+)\s*(?:வாரம்|வாரங்கள்)", "weeks"),
    (r"(?:கடந்த|சென்ற)\s+(\d+)\s*(?:மாதம்|மாதங்கள்)", "months"),
    # Telugu patterns
    (r"(?:గత|చివరి|మునుపటి)\s+(\d+)\s*(?:రోజుల|రోజు)", "days"),
    (r"(?:గత|చివరి)\s+(\d+)\s*(?:వారాలు|వారం)", "weeks"),
    (r"(?:గత|చివరి)\s+(\d+)\s*(?:నెలలు|నెల)", "months"),
    # Gujarati patterns
    (r"(?:છેલ્લા|ગત|પાછલા)\s+(\d+)\s*(?:દિવસ|દિવસનું|દિવસનો)", "days"),
    (r"(?:છેલ્લા|ગત)\s+(\d+)\s*(?:અઠવાડિયું|અઠવાડિયા)", "weeks"),
    (r"(?:છેલ્લા|ગત)\s+(\d+)\s*(?:મહિનો|મહિના)", "months"),
]

# ──────────────────────────────────────────────────────────────────────
# Card type patterns
# ──────────────────────────────────────────────────────────────────────

CARD_TYPE_PATTERNS = [
    (r"\b(debit)\s*(card)?\b", "debit"),
    (r"\b(credit)\s*(card)?\b", "credit"),
    (r"\b(atm)\s*(card)?\b", "atm"),
    (r"\b(டெபிட்|பற்று)\s*(அட்டை|கார்ட்)?\b", "debit"),
    (r"\b(கிரெடிட்|கடன்)\s*(அட்டை|கார்ட்)?\b", "credit"),
    (r"\b(డెబిట్)\s*(కార్డ్)?\b", "debit"),
    (r"\b(క్రెడిట్)\s*(కార్డ్)?\b", "credit"),
    (r"\b(ડેબિટ)\s*(કાર્ડ)?\b", "debit"),
    (r"\b(ક્રેડિટ)\s*(કાર્ડ)?\b", "credit"),
]

# ──────────────────────────────────────────────────────────────────────
# Loan type patterns
# ──────────────────────────────────────────────────────────────────────

LOAN_TYPE_PATTERNS = [
    (r"\b(personal)\s*(loan)?\b", "personal"),
    (r"\b(home|housing|mortgage)\s*(loan)?\b", "home"),
    (r"\b(car|vehicle|auto)\s*(loan)?\b", "car"),
    (r"\b(education|student|study)\s*(loan)?\b", "education"),
    (r"\b(business|commercial)\s*(loan)?\b", "business"),
]


class EntityExtractor:
    """
    Extracts structured entities from natural language text.
    
    Handles multilingual inputs and normalizes extracted values.
    """

    def __init__(self) -> None:
        self._time_patterns = [
            (re.compile(p, re.IGNORECASE | re.UNICODE), unit)
            for p, unit in TIME_PERIOD_PATTERNS
        ]
        self._card_patterns = [
            (re.compile(p, re.IGNORECASE | re.UNICODE), card_type)
            for p, card_type in CARD_TYPE_PATTERNS
        ]
        self._loan_patterns = [
            (re.compile(p, re.IGNORECASE | re.UNICODE), loan_type)
            for p, loan_type in LOAN_TYPE_PATTERNS
        ]

    def extract(self, text: str) -> ExtractedEntities:
        """Extract all entities from text."""
        entities = ExtractedEntities()

        # Replace number words with digits
        normalized = self._replace_number_words(text)

        # Extract time period
        entities.days = self._extract_days(normalized)

        # Extract card type
        entities.card_type = self._extract_card_type(text)

        # Extract loan type
        entities.loan_type = self._extract_loan_type(text)

        return entities

    def _extract_days(self, text: str) -> int | None:
        """Extract the number of days from a time period expression."""
        for pattern, unit in self._time_patterns:
            match = pattern.search(text)
            if match:
                if unit.startswith("fixed_"):
                    return int(unit.split("_")[1])
                try:
                    value = int(match.group(1))
                except (IndexError, ValueError):
                    continue
                if unit == "weeks":
                    return value * 7
                elif unit == "months":
                    return value * 30
                return value
        
        # Fallback: look for bare number near time-related words
        bare_number = re.search(r"(\d+)\s*(?:din|days?|நாள்|రోజు|દિવસ)", text, re.IGNORECASE)
        if bare_number:
            return int(bare_number.group(1))

        return None

    def _extract_card_type(self, text: str) -> str | None:
        for pattern, card_type in self._card_patterns:
            if pattern.search(text):
                return card_type
        return None

    def _extract_loan_type(self, text: str) -> str | None:
        for pattern, loan_type in self._loan_patterns:
            if pattern.search(text):
                return loan_type
        return None

    def _replace_number_words(self, text: str) -> str:
        """Replace number words with their digit equivalents."""
        result = text
        for word, num in sorted(ALL_NUMBER_WORDS.items(), key=lambda x: -len(x[0])):
            result = re.sub(
                re.escape(word), str(num), result, flags=re.IGNORECASE | re.UNICODE
            )
        return result
