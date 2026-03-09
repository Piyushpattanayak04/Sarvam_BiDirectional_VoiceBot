import pathlib, re

path = pathlib.Path('src/nlu/intent_detector.py')
src = path.read_text(encoding='utf-8')

# ── 1. Replace CONFIRMATION + DENIAL patterns ─────────────────────────────────
new_conf = '''    BankingIntent.CONFIRMATION: [
        # English + transliterations — allow trailing punctuation
        r"^\\s*(yes|yeah|yep|sure|ok|okay|confirm|proceed|go ahead|do it|haan|ha|sari|aamaam|bilkul|theek hai|theek|ji haan)\\s*[.!?]?\\s*$",
        # Hindi — core yes-words plus "जी" polite suffix/standalone, with punctuation tolerance
        r"^\\s*(हाँ|हां|हा|जी|बिल्कुल|ठीक|सही)\\s*(जी|है|बिल्कुल)?\\s*[।.!?]?\\s*$",
        r"^\\s*(हाँ जी|जी हाँ|हां जी|जी हां|जी बिल्कुल|हाँ बिल्कुल|ठीक है|बिल्कुल ठीक)\\s*[।.!?]?\\s*$",
        # Other Indian languages — allow trailing punctuation
        r"^\\s*(ஆமாம்|சரி|ஆம்|అవును|సరే|હા|હાં|ઠīk|হ্যাঁ|ঠিক আছে|ହଁ|ଠିକ ଅଛି)\\s*[.!?।]?\\s*$",
    ],
    BankingIntent.DENIAL: [
        # English + transliterations — allow trailing punctuation
        r"^\\s*(no|nope|nah|cancel|stop|don\\'t|never|nahi|venda|vaddu)\\s*[.!?]?\\s*$",
        # Hindi — with optional polite suffix and punctuation tolerance
        r"^\\s*(नहीं|नही|न|मत)\\s*(जी|चाहिए)?\\s*[।.!?]?\\s*$",
        r"^\\s*(नहीं जी|जी नहीं|नहीं चाहिए|बिल्कुल नहीं|नहीं धन्यवाद)\\s*[।.!?]?\\s*$",
        # Other Indian languages — allow trailing punctuation
        r"^\\s*(வேண்டாம்|இல்லை|వద్దు|లేదు|ના|નহीं|না|না ধন্যবাদ|ନା|ଦରকାର ନাহি)\\s*[.!?।]?\\s*$",
    ],'''

old_section = re.search(
    r'    BankingIntent\.CONFIRMATION: \[.*?    BankingIntent\.DENIAL: \[.*?    \],',
    src,
    re.DOTALL
)
if old_section:
    src = src[:old_section.start()] + new_conf + src[old_section.end():]
    print('Patterns replaced OK')
else:
    print('CONFIRMATION/DENIAL section not found - check regex')

# ── 2. Replace LLM_INTENT_SYSTEM_PROMPT ──────────────────────────────────────
new_prompt = r"""LLM_INTENT_SYSTEM_PROMPT = """You are a banking intent classifier. Output ONLY valid JSON.

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
"""

prompt_match = re.search(
    r'LLM_INTENT_SYSTEM_PROMPT = """.*?^}"""',
    src,
    re.DOTALL | re.MULTILINE
)
if prompt_match:
    src = src[:prompt_match.start()] + new_prompt + src[prompt_match.end():]
    print('Prompt replaced OK')
else:
    print('Prompt pattern not found')

path.write_text(src, encoding='utf-8')
print('File written OK')
