"""
Intent Detection Engine

Implements a hybrid intent classification system:

Approach: LLM-first with rule-engine fallback

Why hybrid?
┌────────────────────┬──────────────────┬──────────────────┬────────────────────┐
│ Approach           │ Pros             │ Cons             │ When to use        │
├────────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ Pure LLM           │ Best accuracy,   │ Higher latency   │ Complex/ambiguous  │
│                    │ handles nuance   │ (200-800ms)      │ inputs             │
├────────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ Intent Classifier  │ Fast (<10ms),    │ Needs training   │ High-volume,       │
│ (ML model)         │ predictable      │ data, rigid      │ known intents      │
├────────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ Rule Engine        │ Instant, no API  │ Brittle, can't   │ Keyword-heavy      │
│                    │ dependency       │ handle variation  │ domains            │
├────────────────────┼──────────────────┼──────────────────┼────────────────────┤
│ Hybrid (chosen)    │ Fast for common  │ More complex     │ Production banking │
│                    │ cases, accurate  │ system            │ voice agent        │
│                    │ for edge cases   │                  │                    │
└────────────────────┴──────────────────┴──────────────────┴────────────────────┘

Pipeline:
  1. Rule engine tries keyword matching (< 1ms)
  2. If confidence < threshold → fall back to LLM classification
  3. LLM returns structured JSON with intent + entities
  4. Results cached for identical/similar inputs
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from src.logging_config import get_logger

logger = get_logger(__name__)


class BankingIntent(str, Enum):
    """Supported banking intents."""
    BLOCK_CARD = "block_card"
    BANK_STATEMENT = "bank_statement"
    LOAN_ENQUIRY = "loan_enquiry"
    GREETING = "greeting"
    CONFIRMATION = "confirmation"
    DENIAL = "denial"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    """Result of intent detection."""
    intent: BankingIntent
    confidence: float  # 0.0 to 1.0
    entities: dict[str, str | int | float] = field(default_factory=dict)
    raw_text: str = ""
    detected_language: str = "en"
    method: str = "rule"  # "rule" or "llm"


# ──────────────────────────────────────────────────────────────────────
# Multilingual keyword patterns for rule-based detection
# ──────────────────────────────────────────────────────────────────────

INTENT_PATTERNS: dict[BankingIntent, list[str]] = {
    BankingIntent.BLOCK_CARD: [
        # English
        r"\b(block|freeze|stop|cancel|disable|deactivate)\b.*\b(card|debit|credit|atm)\b",
        r"\b(card)\b.*\b(block|freeze|lost|stolen|missing|compromise)\b",
        r"\b(lost|stolen|missing)\b.*\b(card|debit|credit)\b",
        # Tamil
        r"(அட்டை|கார்ட்).*(தடு|நிறுத்து|முடக்கு|பிளாக்)",
        r"(தடு|நிறுத்து|முடக்கு|பிளாக்).*(அட்டை|கார்ட்)",
        r"(தொலைந்த|தொலைந்தது|காணாமல்|திருடு).*(அட்டை|கார்ட்)",
        r"(அட்டை|கார்ட்).*(தொலைந்த|தொலைந்தது|காணாமல்|திருடு)",
        # Telugu
        r"(కార్డ్|కార్డు).*(బ్లాక్|ఆపు|నిలిపి)",
        r"(బ్లాక్|ఆపు|నిలిపి).*(కార్డ్|కార్డు)",
        r"(పోయిన|కోల్పోయిన).*(కార్డ్|కార్డు)",
        # Gujarati
        r"(કાર્ડ).*(બ્લોક|બંધ|રોક)",
        r"(બ્લોક|બંધ|રોક).*(કાર્ડ)",
        r"(ખોવાયેલ|ગુમ).*(કાર્ડ)",        # Hindi
        r"(कार्ड|कार्ड्).*(ब्लॉक|बंद|रोको|बंद करो)",
        r"(ब्लॉक|बंद करो|रोको).*(कार्ड)",
        r"(कार्ड).*(खो गया|घुम हो गया|चोरी|गुम)",
        r"(खो गया|चोरी|गुम).*(कार्ड)",
        # Bengali
        r"(কার্ড).*(ব্লক|বন্ধ|হারিয়ে|চুরি)",
        r"(ব্লক|বন্ধ করুন).*(কার্ড)",
        r"(কার্ড).*(হারিয়ে গেছে|চুরি হয়েছে)",
        # Odia
        r"(କାର୍ଡ).*(ବ୍ଲକ|ବନ୍ଦ|ହଜି|ଚୋରି)",
        r"(ବ୍ଲକ|ବନ୍ଦ କରନ୍ତୁ).*(କାର୍ଡ)",
        r"(କାର୍ଡ).*(ହଜି ଯାଇଛି|ଚୋରି ହୋଇଛି)",
    ],
    BankingIntent.BANK_STATEMENT: [
        # English
        r"\b(bank|account)\b.*\b(statement|history|transactions?|passbook)\b",
        r"\b(statement|transactions?)\b.*\b(last|past|previous|recent)\b",
        r"\b(last|past|previous)\b.*\b(\d+)\b.*\b(day|week|month)\b",
        r"\b(statement)\b",
        # Tamil
        r"(வங்கி|பேங்க்).*(ஸ்டேட்மென்ட்|அறிக்கை|பரிவர்த்தனை)",
        r"(ஸ்டேட்மென்ட்|அறிக்கை).*(நாட்கள்|நாள்|வாரம்|மாதம்)",
        r"(கடந்த|சென்ற).*(நாட்கள்|நாள்)",
        # Telugu
        r"(బ్యాంక్|ఖాతా).*(స్టేట్మెంట్|లావాదేవీలు)",
        r"(స్టేట్మెంట్).*(రోజుల|రోజు|వారం|నెల)",
        r"(గత|చివరి).*(రోజుల|రోజు)",
        # Gujarati
        r"(બેંક|ખાતા).*(સ્ટેટમેન્ટ|હિસાબ|વ્યવહાર)",
        r"(સ્ટેટમેન્ટ).*(દિવસ|અઠવાડિયું|મહિનો)",
        r"(છેલ્લા|ગત).*(દિવસ)",        # Hindi
        r"(बैंक|खाता).*(स्टेटमेंट|विवरण|लेनदेन)",
        r"(स्टेटमेंट|विवरण).*(दिन|हफ्ते|महीने|दिनों)",
        r"(पिछले|गत|बीते).*(दिन|दिनों|हफ्ते|महीने)",
        # Bengali
        r"(ব্যাংক|অ্যাকাউন্ট).*(স্টেটমেন্ট|লেনদেন|বিবরণী)",
        r"(স্টেটমেন্ট|লেনদেন).*(দিন|সপ্তাহ|মাস)",
        r"(শেষ|গত|বিগত).*(দিন|সপ্তাহ|মাস)",
        # Odia
        r"(ବ୍ୟାଙ୍କ|ଆକାଉଣ୍ଟ).*(ସ୍ଟେଟମେଣ୍ଟ|ଲେଣଦେଣ|ବିବରଣ)",
        r"(ସ୍ଟେଟମେଣ୍ଟ|ଲେଣଦେଣ).*(ଦିନ|ସପ୍ତାହ|ମାସ)",
        r"(ଶେଷ|ଗତ).*(ଦିନ|ସପ୍ତାହ|ମାସ)",
    ],
    BankingIntent.LOAN_ENQUIRY: [
        # English
        r"\b(loans?|emi|borrow|lending|mortgage|credit)\b.*\b(enquir|inquir|ask|know|detail|info|interest|rate|eligib)\b",
        r"\b(enquir|inquir|ask|know|detail|info|eligib)\b.*\b(loans?|emi|borrow|lending|mortgage)\b",
        r"\b(personal|home|car|education|business)\b.*\b(loans?)\b",
        r"\b(loans?)\b",
        # Tamil
        r"(கடன்|லோன்).*(விசாரணை|விவரம்|தகவல்|வட்டி)",
        r"(விசாரணை|விவரம்).*(கடன்|லோன்)",
        # Telugu
        r"(రుణం|లోన్).*(విచారణ|వివరాలు|సమాచారం|వడ్డీ)",
        r"(విచారణ|వివరాలు).*(రుణం|లోన్)",
        # Gujarati
        r"(લોન|ધિરાણ).*(પૂછપરછ|માહિતી|વ્યાજ|વિગત)",
        r"(પૂછપરછ|માહિતી).*(લોન|ધિરાણ)",
        # Hindi
        r"(लोन|ऋण|कर्ज).*(जानना|पूछना|जानकारी|विवरण|ब्याज|दर|चाहिए)",
        r"(जानकारी|पूछना|चाहिए|बारे).*(लोन|ऋण|कर्ज)",
        r"(व्यक्तिगत|होम|गृह|कार|शिक्षा|व्यापार).*(लोन|ऋण)",
        r"(लोन|ऋण|कर्ज)",
        # Bengali
        r"(ঋণ|লোন).*(জানতে|প্রয়োজন|তথ্য|বিবরণ|সুদ)",
        r"(জানতে|পাবা|তথ্য).*(ঋণ|লোন)",
        r"(ব্যক্তিগত|হোম|গৃহ|কার|শিক্ষা).*(ঋণ|লোন)",
        r"(ঋণ|লোন)",
        # Odia
        r"(ଋଣ|ଲୋନ).*(ଜାଣିବା|ସୂଚନା|ବିବରଣ|ଶ୍ରେଣୀ)",
        r"(ଜାଣିବା|ସୂଚନା|ବିଷଯରେ).*(ଋଣ|ଲୋନ)",
        r"(ଋଣ|ଲୋନ)",
    ],
    BankingIntent.GREETING: [
        r"\b(hello|hi|hey|good\s*(morning|afternoon|evening)|namaste|vanakkam)\b",
        r"^(நமஸ்காரம்|வணக்கம்|నమస్కారం|નમસ્તે|नमस्ते|नमस्कार|নমস্কার|হ্যালো|ନମସ୍କାର)$",
    ],
    BankingIntent.CONFIRMATION: [
        # English + transliterations — allow trailing punctuation
        r"^\s*(yes|yeah|yep|sure|ok|okay|confirm|proceed|go ahead|do it|haan|haan ji|ha|sari|aamaam|bilkul|theek hai|theek|ji haan)\s*[.!?]?\s*$",
        # Hindi — core yes-words plus जी polite suffix/standalone, punctuation tolerance
        r"^\s*(हाँ|हां|हा|जी|बिल्कुल|ठीक|सही)\s*(जी|है|बिल्कुल)?\s*[।.!?]?\s*$",
        r"^\s*(हाँ जी|जी हाँ|हां जी|जी हां|जी बिल्कुल|हाँ बिल्कुल|ठीक है|बिल्कुल ठीक)\s*[।.!?]?\s*$",
        # Other Indian languages — allow trailing punctuation
        r"^\s*(ஆமாம்|சரி|ஆம்|అవును|సరే|ગા|ગાં|ગુજ|হ্যাঁ|ঠিক আছে|ୱ|ଠିକ ଅଛି)\s*[.!?।]?\s*$",
    ],
    BankingIntent.DENIAL: [
        # English + transliterations — allow trailing punctuation
        r"^\s*(no|nope|nah|cancel|stop|don't|never|nahi|nahi ji|venda|vaddu)\s*[.!?]?\s*$",
        # Hindi — with optional polite suffix and punctuation tolerance
        r"^\s*(नहीं|नही|न|मत)\s*(जी|चाहिए)?\s*[।.!?]?\s*$",
        r"^\s*(नहीं जी|जी नहीं|नहीं चाहिए|बिल्कुल नहीं|नहीं धन्यवाद)\s*[।.!?]?\s*$",
        # Other Indian languages — allow trailing punctuation
        r"^\s*(வேண்டாம்|இல்லை|వద్దు|నేదు|ના|નહીં|না|না ধন্যবাদ|ନା|ଦରକାର ନାହି)\s*[.!?।]?\s*$",
    ],
}


class RuleBasedIntentDetector:
    """
    Fast keyword/regex-based intent detector.
    Runs in <1ms, used as first pass before LLM fallback.
    """

    def __init__(self) -> None:
        # Pre-compile all patterns
        self._compiled_patterns: dict[BankingIntent, list[re.Pattern[str]]] = {}
        for intent, patterns in INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns
            ]

    def detect(self, text: str) -> IntentResult:
        """
        Detect intent from text using regex patterns.
        Returns the highest-confidence match.
        """
        text = text.strip()
        if not text:
            return IntentResult(
                intent=BankingIntent.UNKNOWN, confidence=0.0, raw_text=text
            )

        best_intent = BankingIntent.UNKNOWN
        best_confidence = 0.0
        match_count = 0

        for intent, patterns in self._compiled_patterns.items():
            hits = sum(1 for p in patterns if p.search(text))
            if hits > 0:
                # Confidence based on proportion of patterns matched
                confidence = min(0.5 + (hits / len(patterns)) * 0.5, 0.95)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_intent = intent
                    match_count = hits

        return IntentResult(
            intent=best_intent,
            confidence=best_confidence,
            raw_text=text,
            method="rule",
        )


# ──────────────────────────────────────────────────────────────────────
# LLM-based intent detection prompt
# ──────────────────────────────────────────────────────────────────────

LLM_INTENT_SYSTEM_PROMPT = """You are a banking intent classifier. Output ONLY valid JSON.

Intents:
- block_card: block/freeze/lost/stolen card
- bank_statement: statement/history/transactions/passbook
- loan_enquiry: loans/EMI/interest/eligibility/borrow
- greeting: hello/hi/namaste
- confirmation: yes/ok/sure/haan/haan ji
- denial: no/cancel/stop/nahi
- unknown: ONLY if no banking context whatsoever

Rules:
- NEVER default to unknown if any banking context exists.
- Vague intent in banking conversation -> loan_enquiry.
- Card lost/stolen/freeze -> block_card.
- Transaction history/passbook -> bank_statement.
- Loan/EMI/interest/borrow -> loan_enquiry.

Extract: "days" (int) for bank_statement, "loan_type" (str) for loan_enquiry.
Detect language code: en/hi/ta/te/gu/bn/or/mixed.

{"intent":"...","confidence":0.0-1.0,"entities":{},"detected_language":"..."}"""


def build_llm_intent_prompt(user_text: str, conversation_context: str = "") -> list[dict[str, str]]:
    """Build the messages array for LLM intent classification."""
    messages = [
        {"role": "system", "content": LLM_INTENT_SYSTEM_PROMPT},
    ]
    if conversation_context:
        messages.append({
            "role": "user",
            "content": f"Previous conversation context:\n{conversation_context}",
        })
    messages.append({"role": "user", "content": f"Classify this user utterance:\n\"{user_text}\""})
    return messages


def parse_llm_intent_response(response_text: str, raw_text: str) -> IntentResult:
    """Parse LLM JSON response into IntentResult."""
    try:
        # Extract the outermost JSON object from the response.
        # Use a greedy match (.*) so nested objects (e.g. entities dict) are included.
        # The previous non-greedy [^{}]* pattern broke on responses containing
        # nested braces like {"entities": {"loan_type": "personal"}}.
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in LLM response")
        
        data = json.loads(json_match.group())
        
        intent_str = data.get("intent", "unknown")
        try:
            intent = BankingIntent(intent_str)
        except ValueError:
            intent = BankingIntent.UNKNOWN

        return IntentResult(
            intent=intent,
            confidence=float(data.get("confidence", 0.5)),
            entities=data.get("entities", {}),
            raw_text=raw_text,
            detected_language=data.get("detected_language", "en"),
            method="llm",
        )
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("llm_intent_parse_error", error=str(e), response=response_text)
        return IntentResult(
            intent=BankingIntent.UNKNOWN,
            confidence=0.0,
            raw_text=raw_text,
            method="llm",
        )
