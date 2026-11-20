from __future__ import annotations

import uuid
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentiq.chat.sql_models import ChatMessageRecord, ConversationRecord


class ConversationNotFoundError(Exception):
    """Raised when a chat conversation does not exist."""


class ChatRepository:
    """PostgreSQL repository for chat conversations and messages."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_conversation(self) -> ConversationRecord:
        now = datetime.now(timezone.utc)

        conversation = ConversationRecord(
            id=uuid.uuid4(),
            created_at=now,
            updated_at=now,
        )

        self._session.add(conversation)
        await self._session.flush()

        return conversation

    async def get_conversation(
        self,
        conversation_id: UUID,
    ) -> ConversationRecord:
        conversation = await self._session.get(
            ConversationRecord,
            conversation_id,
        )

        if conversation is None:
            raise ConversationNotFoundError(
                f"Chat conversation not found: {conversation_id}"
            )

        return conversation

    async def append_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        *,
        message_id: UUID | None = None,
        truncated: bool = False,
    ) -> ChatMessageRecord:
        await self.get_conversation(conversation_id)

        now = datetime.now(timezone.utc)

        message = ChatMessageRecord(
            id=message_id or uuid.uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            truncated=truncated,
            created_at=now,
        )

        self._session.add(message)

        conversation = await self.get_conversation(conversation_id)
        conversation.updated_at = now

        await self._session.flush()

        return message

    async def update_message(
        self,
        message_id: UUID,
        *,
        content: str,
        truncated: bool,
    ) -> ChatMessageRecord:
        message = await self._session.get(ChatMessageRecord, message_id)

        if message is None:
            raise ConversationNotFoundError(f"Chat message not found: {message_id}")

        message.content = content
        message.truncated = truncated

        conversation = await self.get_conversation(message.conversation_id)
        conversation.updated_at = datetime.now(timezone.utc)

        await self._session.flush()

        return message

    async def get_messages(
        self,
        conversation_id: UUID,
    ) -> list[ChatMessageRecord]:
        await self.get_conversation(conversation_id)

        result = await self._session.execute(
            select(ChatMessageRecord)
            .where(ChatMessageRecord.conversation_id == conversation_id)
            .order_by(ChatMessageRecord.created_at)
        )

        return list(result.scalars().all())

    async def get_message(
        self,
        message_id: UUID,
    ) -> ChatMessageRecord:
        message = await self._session.get(ChatMessageRecord, message_id)

        if message is None:
            raise ConversationNotFoundError(f"Chat message not found: {message_id}")

        return message
