"""
Dialogue State Machine

Manages the conversation flow for banking interactions using a finite state model.

State transitions:
    IDLE → LISTENING → PROCESSING → AWAITING_CONFIRMATION → EXECUTING → RESPONDING → LISTENING
                ↑                           ↓
                └───────── CLARIFYING ←──────┘

Each state has:
  - Entry actions (what to do when entering)
  - Valid transitions (which states can follow)
  - Timeout behavior (what happens if no input)

Design: State is per-session and stored in-memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DialogueState(str, Enum):
    """States in the conversation flow."""
    IDLE = "idle"                           # No active interaction
    GREETING = "greeting"                   # Initial greeting
    LISTENING = "listening"                 # Actively listening to user
    PROCESSING = "processing"              # Processing user input
    CLARIFYING = "clarifying"              # Asking for clarification
    AWAITING_CONFIRMATION = "awaiting_confirmation"  # Waiting for yes/no
    EXECUTING = "executing"                # Calling banking API
    RESPONDING = "responding"              # Speaking response to user
    ERROR = "error"                        # Error state
    ENDED = "ended"                        # Conversation ended


# Valid state transitions
VALID_TRANSITIONS: dict[DialogueState, set[DialogueState]] = {
    DialogueState.IDLE: {DialogueState.GREETING, DialogueState.LISTENING},
    DialogueState.GREETING: {DialogueState.LISTENING},
    DialogueState.LISTENING: {DialogueState.PROCESSING, DialogueState.ENDED},
    DialogueState.PROCESSING: {
        DialogueState.CLARIFYING,
        DialogueState.AWAITING_CONFIRMATION,
        DialogueState.EXECUTING,
        DialogueState.RESPONDING,
        DialogueState.ERROR,
    },
    DialogueState.CLARIFYING: {DialogueState.LISTENING, DialogueState.ENDED},
    DialogueState.AWAITING_CONFIRMATION: {
        DialogueState.EXECUTING,
        DialogueState.LISTENING,
        DialogueState.ENDED,
    },
    DialogueState.EXECUTING: {DialogueState.RESPONDING, DialogueState.ERROR},
    DialogueState.RESPONDING: {DialogueState.LISTENING, DialogueState.ENDED},
    DialogueState.ERROR: {DialogueState.LISTENING, DialogueState.RESPONDING, DialogueState.ENDED},
    DialogueState.ENDED: set(),
}


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    text: str
    intent: str = ""
    entities: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: float = 0.0


@dataclass
class ConversationContext:
    """
    Full conversation context for a single session.
    
    This is the primary state object passed between pipeline stages.
    Stored in-memory for persistence during the session.
    """
    session_id: str
    state: DialogueState = DialogueState.IDLE
    turns: list[ConversationTurn] = field(default_factory=list)
    
    # Current intent being processed
    current_intent: str = ""
    current_entities: dict[str, Any] = field(default_factory=dict)
    
    # Pending clarification
    pending_slot: str = ""  # Which slot we're asking about (e.g., "days", "card_type")
    
    # User preferences
    detected_language: str = "en"
    preferred_language: str = "en-IN"
    
    # Metadata
    user_id: str = ""
    created_at_ms: float = 0.0
    last_activity_ms: float = 0.0
    turn_count: int = 0

    def add_turn(self, role: str, text: str, **kwargs: Any) -> None:
        """Add a conversation turn."""
        self.turns.append(ConversationTurn(role=role, text=text, **kwargs))
        self.turn_count += 1

    def get_context_summary(self, max_turns: int = 6) -> str:
        """Get a summary of recent conversation for LLM context."""
        recent = self.turns[-max_turns:]
        return "\n".join(f"{t.role}: {t.text}" for t in recent)

    def transition_to(self, new_state: DialogueState) -> bool:
        """
        Attempt a state transition. Returns True if valid.
        
        Enforces the state machine to prevent invalid transitions.
        """
        if new_state in VALID_TRANSITIONS.get(self.state, set()):
            self.state = new_state
            return True
        return False

    def merge_entities(self, new_entities: dict[str, Any]) -> None:
        """Merge new entities into current context, preserving existing values."""
        for key, value in new_entities.items():
            if value is not None:
                self.current_entities[key] = value
