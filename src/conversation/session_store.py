"""Session Store — In-memory session persistence for conversation context."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from src.conversation.state import ConversationContext, DialogueState
from src.logging_config import get_logger

logger = get_logger(__name__)


class SessionStore(ABC):
    """Abstract session store interface."""

    @abstractmethod
    async def get(self, session_id: str) -> ConversationContext | None:
        ...

    @abstractmethod
    async def save(self, context: ConversationContext) -> None:
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        ...


class InMemorySessionStore(SessionStore):
    """In-memory session store for development/testing."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def get(self, session_id: str) -> ConversationContext | None:
        data = self._store.get(session_id)
        if data is None:
            return None
        return self._deserialize(data)

    async def save(self, context: ConversationContext) -> None:
        context.last_activity_ms = time.time() * 1000
        self._store[context.session_id] = self._serialize(context)

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    async def exists(self, session_id: str) -> bool:
        return session_id in self._store

    def _serialize(self, context: ConversationContext) -> dict[str, Any]:
        return {
            "session_id": context.session_id,
            "state": context.state.value,
            "turns": [
                {
                    "role": t.role,
                    "text": t.text,
                    "intent": t.intent,
                    "entities": t.entities,
                    "timestamp_ms": t.timestamp_ms,
                }
                for t in context.turns
            ],
            "current_intent": context.current_intent,
            "current_entities": context.current_entities,
            "pending_slot": context.pending_slot,
            "detected_language": context.detected_language,
            "preferred_language": context.preferred_language,
            "user_id": context.user_id,
            "created_at_ms": context.created_at_ms,
            "last_activity_ms": context.last_activity_ms,
            "turn_count": context.turn_count,
        }

    def _deserialize(self, data: dict[str, Any]) -> ConversationContext:
        from src.conversation.state import ConversationTurn

        ctx = ConversationContext(
            session_id=data["session_id"],
            state=DialogueState(data["state"]),
            current_intent=data.get("current_intent", ""),
            current_entities=data.get("current_entities", {}),
            pending_slot=data.get("pending_slot", ""),
            detected_language=data.get("detected_language", "en"),
            preferred_language=data.get("preferred_language", "en-IN"),
            user_id=data.get("user_id", ""),
            created_at_ms=data.get("created_at_ms", 0.0),
            last_activity_ms=data.get("last_activity_ms", 0.0),
            turn_count=data.get("turn_count", 0),
        )
        for t in data.get("turns", []):
            ctx.turns.append(ConversationTurn(**t))
        return ctx
