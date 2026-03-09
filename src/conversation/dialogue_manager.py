"""
Dialogue Manager

The central brain of the conversation system. Receives NLU results,
manages state transitions, generates appropriate responses, and 
coordinates with the banking API layer.

Architecture:
    Audio Pipeline → STT → NLU Pipeline
                                ↓
                        Dialogue Manager
                        ├── State Machine
                        ├── Response Generator
                        ├── Banking API Client
                        └── Context Store (In-Memory)
                                ↓
                        TTS → Audio Pipeline
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.conversation.state import ConversationContext, DialogueState
from src.logging_config import get_logger
from src.nlu.intent_detector import BankingIntent
from src.nlu.pipeline import NLUResult

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Response templates (multilingual)
# ──────────────────────────────────────────────────────────────────────

RESPONSE_TEMPLATES: dict[str, dict[str, str]] = {
    "greeting": {
        "en-IN": "Welcome to our banking assistant. How can I help you today? You can ask me to block a card, get a bank statement, or enquire about loans.",
        "hi-IN": "हमारे बैंकिंग सहायक में आपका स्वागत है। आज मैं आपकी कैसे मदद कर सकता हूँ? आप कार्ड ब्लॉक करवाने, बैंक स्टेटमेंट लेने, या लोन के बारे में पूछ सकते हैं।",
        "ta-IN": "எங்கள் வங்கி உதவியாளருக்கு வரவேற்கிறோம். இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        "te-IN": "మా బ్యాంకింగ్ అసిస్టెంట్‌కు స్వాగతం. ఈ రోజు నేను మీకు ఎలా సహాయం చేయగలను?",
        "gu-IN": "અમારા બેંકિંગ સહાયકમાં સ્વાગત છે. આજે હું તમને કેવી રીતે મદદ કરી શકું?",
        "bn-IN": "আমাদের ব্যাংকিং সহকারীতে আপনাকে স্বাগতম। আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি? আপনি কার্ড ব্লক করতে, ব্যাংক স্টেটমেন্ট পেতে, বা ঋণ সম্পর্কে জানতে চাইতে পারেন।",
        "or-IN": "ଆମର ବ୍ୟାଙ୍କିଂ ସହାୟକକୁ ଆପଣଙ୍କୁ ସ୍ୟାଗତମ। ଆଜି ମୁଁ ଆପଣଙ୍କୁ କିଭଳି ସାହାୟ୍ୟ କରିପାରିବି? ଆପଣୀ କାର୍ଡ ବ୍ଲକ, ବ୍ୟାଙ୍କ ସ୍ଟେଟମେଣ୍ଟ, ବା ଋଣ ବିଷଯରେ ପଚାରି ପାରିବେ।",
    },
    "block_card_confirm": {
        "en-IN": "I'll block your {card_type} card right away. Can you confirm you want to proceed?",
        "hi-IN": "मैं अभी आपका {card_type} कार्ड ब्लॉक कर देता हूँ। क्या आप आगे बढ़ना चाहते हैं?",
        "ta-IN": "உங்கள் {card_type} அட்டையை உடனடியாக தடுக்கிறேன். தொடர விரும்புகிறீர்களா?",
        "te-IN": "మీ {card_type} కార్డ్‌ను వెంటనే బ్లాక్ చేస్తాను. మీరు కొనసాగించాలనుకుంటున్నారా?",
        "gu-IN": "હું તમારું {card_type} કાર્ડ તરત જ બ્લોક કરીશ. શું તમે આગળ વધવા માંગો છો?",
        "bn-IN": "আমি এখনই আপনার {card_type} কার্ড ব্লক করে দিচ্ছি। আপনি কি এগিয়ে যেতে চান?",
        "or-IN": "ମୁଁ ଏବେ ଆପଣଙ୍କର {card_type} କାର୍ଡ ବ୍ଲକ କରିଦେବି। ଆପଣୀ କି ଆଗେଜିବାକୁ ଚାହୁଁଛନ୍ତି?",
    },
    "block_card_success": {
        "en-IN": "Your {card_type} card has been blocked successfully. A new card will be sent to your registered address within 5-7 business days.",
        "hi-IN": "आपका {card_type} कार्ड सफलतापूर्वक ब्लॉक हो गया है। नया कार्ड 5-7 कार्य दिवसों में आपके पंजीकृत पते पर भेजा जाएगा।",
        "ta-IN": "உங்கள் {card_type} அட்டை வெற்றிகரமாக தடுக்கப்பட்டது. புதிய அட்டை 5-7 வணிக நாட்களில் அனுப்பப்படும்.",
        "te-IN": "మీ {card_type} కార్డ్ విజయవంతంగా బ్లాక్ చేయబడింది. కొత్త కార్డ్ 5-7 వ్యాపార రోజులలో పంపబడుతుంది.",
        "gu-IN": "તમારું {card_type} કાર્ડ સફળતાપૂર્વક બ્લોક થઈ ગયું છે. નવું કાર્ડ 5-7 કાર્ય દિવસોમાં મોકલવામાં આવશે.",        "bn-IN": "আপনার {card_type} কার্ড সফলভাবে ব্লক হয়ে গেছে। নতুন কার্ড 5-7 কার্যদিবসের মধ্যে আপনার নিবন্ধিত ঠিকানায় পাঠানো হবে।",
        "or-IN": "ଆପଣଙ୍କର {card_type} କାର୍ଡ ସଫଳତାର ସହ ବ୍ଲକ ହୋଇଗଲା। ନୂଆ କାର୍ଡ 5-7 କାର୍ୟ ଦିବସ ମଧ୍ୟରେ ଆପଣଙ୍କ ଠିକଣାକୁ ପଚାରା ଯାଇବ।",    },
    "bank_statement_ask_days": {
        "en-IN": "For how many days would you like the bank statement?",
        "hi-IN": "आप कितने दिनों का बैंक स्टेटमेंट चाहते हैं?",
        "ta-IN": "எத்தனை நாட்களுக்கு வங்கி அறிக்கை வேண்டும்?",
        "te-IN": "ఎన్ని రోజుల బ్యాంక్ స్టేట్‌మెంట్ కావాలి?",
        "gu-IN": "કેટલા દિવસનું બેંક સ્ટેટમેન્ટ જોઈએ?",
        "bn-IN": "আপনি কত দিনের ব্যাংক স্টেটমেন্ট চান?",
        "or-IN": "ଆପଣୀ କେତେ ଦିନର ବ୍ୟାଙ୍କ ସ୍ଟେଟମେଣ୍ଟ ଚାହୁଁଛନ୍ତି?",
    },
    "bank_statement_success": {
        "en-IN": "Your bank statement for the last {days} days has been generated and sent to your registered email address.",
        "hi-IN": "पिछले {days} दिनों का आपका बैंक स्टेटमेंट तैयार हो गया है और आपके पंजीकृत ईमेल पर भेज दिया गया है।",
        "ta-IN": "கடந்த {days} நாட்களுக்கான உங்கள் வங்கி அறிக்கை உருவாக்கப்பட்டு பதிவு செய்யப்பட்ட மின்னஞ்சலுக்கு அனுப்பப்பட்டது.",
        "te-IN": "గత {days} రోజుల మీ బ్యాంక్ స్టేట్‌మెంట్ రూపొందించబడింది మరియు మీ నమోదిత ఇమెయిల్‌కు పంపబడింది.",
        "gu-IN": "છેલ્લા {days} દિવસનું તમારું બેંક સ્ટેટમેન્ટ તૈયાર કરીને તમારા રજિસ્ટર્ડ ઈમેલ પર મોકલવામાં આવ્યું છે.",        "bn-IN": "গত {days} দিনের আপনার ব্যাংক স্টেটমেন্ট তৈরি হয়েছে এবং আপনার নিবন্ধিত ইমেইলে পাঠানো হয়েছে।",
        "or-IN": "ଗତ {days} ଦିନର ଆପଣଙ୍କର ବ୍ୟାଙ୍କ ସ୍ଟେଟମେଣ୍ଟ ପ୍ରସ୍ତୁତ ହୋଇ ଆପଣଙ୍କ ଇ-ମେଲ୍ ରେ ପଠାଯାଇଛି।",    },
    "loan_enquiry_response": {
        "en-IN": "We offer {loan_type} loans starting from 10.5% annual interest rate. The loan amount ranges from 50,000 to 50 lakhs. Would you like me to check your eligibility?",
        "hi-IN": "हम {loan_type} लोन 10.5% वार्षिक ब्याज दर से प्रदान करते हैं। लोन की राशि 50,000 से 50 लाख तक है। क्या आप अपनी पात्रता जाँचना चाहते हैं?",
        "ta-IN": "{loan_type} கடன்களை 10.5% வருடாந்திர வட்டி விகிதத்தில் வழங்குகிறோம். கடன் தொகை 50,000 முதல் 50 லட்சம் வரை. தகுதியை சரிபார்க்க வேண்டுமா?",
        "te-IN": "మేము {loan_type} రుణాలను 10.5% వార్షిక వడ్డీ రేటుతో అందిస్తున్నాము. రుణ మొత్తం 50,000 నుండి 50 లక్షల వరకు. మీ అర్హతను తనిఖీ చేయమంటారా?",
        "gu-IN": "અમે {loan_type} લોન 10.5% વાર્ષિક વ્યાજ દરથી ઓફર કરીએ છીએ. લોનની રકમ 50,000 થી 50 લાખ સુધી છે. શું તમે તમારી પાત્રતા તપાસવા માંગો છો?",        "bn-IN": "আমরা {loan_type} ঋণ 10.5% বার্ষিক সুদের হারে প্রদান করি। ঋণের পরিমাণ 50,000 থেকে 50 লাখ পর্যন্ত। আপনি কি যোগ্যতা যাচাই করতে চান?",
        "or-IN": "ଆମେ {loan_type} ଋଣ 10.5% ବାର୍ଷିକ ସୁଧ ହାରରେ ପ୍ରଦାନ କରୁଁ। ଋଣ ପରିମାଣ 50,000 ରୁ 50 ଲକ୍ଷ ପର୍ୟନ୍ତ। ଆପଣୀ କି ୟୋଗ୍ୟତା ଯାଚନା କରିବାକୁ ଚାହୁଁଛନ୍ତି?",    },
    "error": {
        "en-IN": "I'm sorry, something went wrong. Could you please try again?",
        "hi-IN": "माफ करें, कुछ गड़बड़ हो गई। क्या आप फिर से कोशिश कर सकते हैं?",
        "ta-IN": "மன்னிக்கவும், ஏதோ தவறு ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.",
        "te-IN": "క్షమించండి, ఏదో తప్పు జరిగింది. దయచేసి మళ్ళీ ప్రయత్నించండి.",
        "gu-IN": "માફ કરશો, કંઈક ખોટું થયું. કૃપા કરીને ફરીથી પ્રયાસ કરો.",
        "bn-IN": "দুঃখিত, কিছু একটা ভুল হয়ে গেছে। অনুগ্রহ করে আবার চেষ্টা করুন।",
        "or-IN": "ଦୁଣ୍ଖିତ, କିଛି ଭୁଲ ହୋଇଗଲା। ଦୟାକରି ପୁଣି ଚେଷ୍ଟା କରନ୍ତୁ।",
    },
    "anything_else": {
        "en-IN": "Is there anything else I can help you with?",
        "hi-IN": "क्या मैं आपकी और कोई मदद कर सकता हूँ?",
        "ta-IN": "வேறு ஏதாவது உதவி வேண்டுமா?",
        "te-IN": "మీకు ఇంకేదైనా సహాయం కావాలా?",
        "gu-IN": "શું તમને બીજું કંઈ મદદ જોઈએ?",
        "bn-IN": "আর কোনো সাহায্য প্রয়োজন?",
        "or-IN": "ଆଉ କିଛି ସାହାୟ୍ୟ ଦରକାର?",
    },
    "not_understood": {
        "en-IN": "I'm sorry, I didn't quite understand. Could you please tell me if you'd like to block a card, get a bank statement, or enquire about a loan?",
        "hi-IN": "माफ करें, मुझे समझ नहीं आया। क्या आप कार्ड ब्लॉक करवाना चाहते हैं, बैंक स्टेटमेंट लेना चाहते हैं, या लोन के बारे में जानना चाहते हैं?",
        "ta-IN": "மன்னிக்கவும், புரியவில்லை. அட்டை தடுக்க, வங்கி அறிக்கை பெற, அல்லது கடன் விசாரிக்க விரும்புகிறீர்களா?",
        "te-IN": "క్షమించండి, అర్థం కాలేదు. కార్డ్ బ్లాక్ చేయాలా, బ్యాంక్ స్టేట్‌మెంట్ కావాలా, లేదా లోన్ గురించి తెలుసుకోవాలా?",
        "gu-IN": "માફ કરશો, સમજ ન પડ્યું. કાર્ડ બ્લોક કરવું છે, બેંક સ્ટેટમેન્ટ જોઈએ, કે લોન વિશે જાણવું છે?",
        "bn-IN": "দুঃখিত, আমি বুঝতে পারিনি। আপনি কি কার্ড ব্লক করতে, ব্যাংক স্টেটমেন্ট পেতে, বা ঋণ সম্পর্কে জানতে চান?",
        "or-IN": "ଦୁଣ୍ଖିତ, ବୁଝିଲିନି। ଆପଣୀ କି କାର୍ଡ ବ୍ଲକ, ବ୍ୟାଙ୍କ ସ୍ଟେଟମେଣ୍ଟ, ବା ଋଣ ବିଷଯରେ ଜାଣିବାକୁ ଚାହୁଁଛନ୍ତି?",
    },
}


@dataclass
class DialogueAction:
    """Action to be taken by the pipeline after dialogue processing."""
    response_text: str
    next_state: DialogueState
    should_execute_api: bool = False
    api_action: str = ""
    api_params: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.api_params is None:
            self.api_params = {}


class DialogueManager:
    """
    Orchestrates conversation flow based on NLU results and current state.
    
    This is a stateless processor — all state lives in ConversationContext.
    This allows horizontal scaling: any server instance can handle any session.
    """

    def get_response_template(self, key: str, lang: str, **kwargs: str | int) -> str:
        """Get a localized response template with variable substitution."""
        templates = RESPONSE_TEMPLATES.get(key, {})
        template = templates.get(lang, templates.get("en-IN", ""))
        try:
            return template.format(**kwargs)
        except KeyError:
            return template

    async def process_turn(
        self,
        nlu_result: NLUResult,
        context: ConversationContext,
    ) -> DialogueAction:
        """
        Process a single conversation turn.
        
        Takes the NLU result and current context, determines the
        appropriate response and state transition.
        """
        lang = context.preferred_language
        intent = nlu_result.intent.intent
        entities = nlu_result.entities

        # Save the previous intent before overwriting (needed for confirmation handling)
        previous_intent = context.current_intent

        # Update context with new NLU result
        # Only overwrite current_intent for non-confirmation/denial intents
        if intent not in (BankingIntent.CONFIRMATION, BankingIntent.DENIAL):
            context.current_intent = intent.value
        context.merge_entities(
            {
                "days": entities.days,
                "card_type": entities.card_type,
                "loan_type": entities.loan_type,
            }
        )

        logger.info(
            "dialogue_processing",
            state=context.state.value,
            intent=intent.value,
            entities=context.current_entities,
        )

        # ── Handle based on current state ────────────────────────────
        
        if context.state == DialogueState.IDLE:
            context.transition_to(DialogueState.GREETING)
            response = self.get_response_template("greeting", lang)
            return DialogueAction(
                response_text=response,
                next_state=DialogueState.LISTENING,
            )

        if context.state == DialogueState.AWAITING_CONFIRMATION:
            return await self._handle_confirmation(intent, context, lang)

        if context.state == DialogueState.CLARIFYING:
            return await self._handle_clarification(nlu_result, context, lang)

        # ── Handle by intent ─────────────────────────────────────────

        match intent:
            case BankingIntent.GREETING:
                response = self.get_response_template("greeting", lang)
                return DialogueAction(
                    response_text=response,
                    next_state=DialogueState.LISTENING,
                )

            case BankingIntent.BLOCK_CARD:
                return await self._handle_block_card(context, lang)

            case BankingIntent.BANK_STATEMENT:
                return await self._handle_bank_statement(context, lang)

            case BankingIntent.LOAN_ENQUIRY:
                return await self._handle_loan_enquiry(context, lang)

            case BankingIntent.UNKNOWN:
                if nlu_result.requires_clarification:
                    context.transition_to(DialogueState.CLARIFYING)
                    return DialogueAction(
                        response_text=self.get_response_template("not_understood", lang),
                        next_state=DialogueState.LISTENING,
                    )
                response = self.get_response_template("error", lang)
                return DialogueAction(
                    response_text=response,
                    next_state=DialogueState.LISTENING,
                )

            case _:
                response = self.get_response_template("error", lang)
                return DialogueAction(
                    response_text=response,
                    next_state=DialogueState.LISTENING,
                )

    async def _handle_block_card(
        self, context: ConversationContext, lang: str
    ) -> DialogueAction:
        """Handle block card intent flow."""
        card_type = context.current_entities.get("card_type", "debit")
        
        if not context.current_entities.get("card_type"):
            # Ask which card
            context.transition_to(DialogueState.CLARIFYING)
            context.pending_slot = "card_type"
            return DialogueAction(
                response_text=self.get_response_template(
                    "block_card_confirm", lang, card_type="debit"
                ),
                next_state=DialogueState.LISTENING,
            )

        # Ask for confirmation
        context.transition_to(DialogueState.AWAITING_CONFIRMATION)
        return DialogueAction(
            response_text=self.get_response_template(
                "block_card_confirm", lang, card_type=card_type
            ),
            next_state=DialogueState.AWAITING_CONFIRMATION,
        )

    async def _handle_bank_statement(
        self, context: ConversationContext, lang: str
    ) -> DialogueAction:
        """Handle bank statement intent flow."""
        days = context.current_entities.get("days")

        if days is None:
            context.transition_to(DialogueState.CLARIFYING)
            context.pending_slot = "days"
            return DialogueAction(
                response_text=self.get_response_template("bank_statement_ask_days", lang),
                next_state=DialogueState.LISTENING,
            )

        # We have all we need — execute
        context.transition_to(DialogueState.EXECUTING)
        return DialogueAction(
            response_text=self.get_response_template(
                "bank_statement_success", lang, days=str(days)
            ),
            next_state=DialogueState.LISTENING,
            should_execute_api=True,
            api_action="bank_statement",
            api_params={"days": days, "user_id": context.user_id},
        )

    async def _handle_loan_enquiry(
        self, context: ConversationContext, lang: str
    ) -> DialogueAction:
        """Handle loan enquiry intent flow."""
        loan_type = context.current_entities.get("loan_type", "personal")

        return DialogueAction(
            response_text=self.get_response_template(
                "loan_enquiry_response", lang, loan_type=loan_type
            ),
            next_state=DialogueState.LISTENING,
        )

    async def _handle_confirmation(
        self,
        intent: BankingIntent,
        context: ConversationContext,
        lang: str,
    ) -> DialogueAction:
        """Handle yes/no confirmation."""
        if intent == BankingIntent.CONFIRMATION:
            # Execute the pending action
            if context.current_intent == BankingIntent.BLOCK_CARD.value:
                card_type = context.current_entities.get("card_type", "debit")
                context.transition_to(DialogueState.EXECUTING)
                return DialogueAction(
                    response_text=self.get_response_template(
                        "block_card_success", lang, card_type=card_type
                    ),
                    next_state=DialogueState.LISTENING,
                    should_execute_api=True,
                    api_action="block_card",
                    api_params={
                        "card_type": card_type,
                        "user_id": context.user_id,
                    },
                )
            # Generic confirmation
            context.transition_to(DialogueState.LISTENING)
            return DialogueAction(
                response_text=self.get_response_template("anything_else", lang),
                next_state=DialogueState.LISTENING,
            )

        elif intent == BankingIntent.DENIAL:
            context.transition_to(DialogueState.LISTENING)
            return DialogueAction(
                response_text=self.get_response_template("anything_else", lang),
                next_state=DialogueState.LISTENING,
            )

        else:
            # User changed topic — process as new intent
            context.transition_to(DialogueState.PROCESSING)
            # Return a signal that we need to re-process
            return DialogueAction(
                response_text="",
                next_state=DialogueState.PROCESSING,
            )

    async def _handle_clarification(
        self,
        nlu_result: NLUResult,
        context: ConversationContext,
        lang: str,
    ) -> DialogueAction:
        """Handle response to a clarification question."""
        slot = context.pending_slot
        entities = nlu_result.entities

        if slot == "days" and entities.days is not None:
            context.current_entities["days"] = entities.days
            context.pending_slot = ""
            return await self._handle_bank_statement(context, lang)

        if slot == "card_type" and entities.card_type is not None:
            context.current_entities["card_type"] = entities.card_type
            context.pending_slot = ""
            return await self._handle_block_card(context, lang)

        # Still couldn't extract — ask again or process as new intent
        if nlu_result.intent.intent != BankingIntent.UNKNOWN:
            # User changed intent
            context.pending_slot = ""
            context.transition_to(DialogueState.PROCESSING)
            return DialogueAction(
                response_text="", next_state=DialogueState.PROCESSING
            )

        return DialogueAction(
            response_text=self.get_response_template("error", lang),
            next_state=DialogueState.LISTENING,
        )
