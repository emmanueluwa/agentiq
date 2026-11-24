from __future__ import annotations

import uuid
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentiq.chat.sql_models import (
    ChatMessageRecord,
    ConversationRecord,
    ChatMessageSourceRecord,
)


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
        cited_source_ids: list[str] | None = None,
        invalid_source_ids: list[str] | None = None,
        grounded: bool | None = None,
        sources: list[dict] | None = None,
    ) -> ChatMessageRecord:
        message = await self._session.get(ChatMessageRecord, message_id)

        if message is None:
            raise ConversationNotFoundError(f"Chat message not found: {message_id}")

        message.content = content
        message.truncated = truncated

        if cited_source_ids is not None:
            message.cited_source_ids = cited_source_ids

        if invalid_source_ids is not None:
            message.invalid_source_ids = invalid_source_ids

        if grounded is not None:
            message.grounded = grounded

        if sources is not None:
            await self._session.execute(
                delete(ChatMessageSourceRecord).where(
                    ChatMessageSourceRecord.message_id == message.id,
                )
            )

            if sources:
                self._session.add_all(
                    [
                        ChatMessageSourceRecord(
                            id=uuid.uuid4(),
                            message_id=message.id,
                            source_number=source["source_number"],
                            source_id=source["source_id"],
                            title=source["title"],
                            url=source["url"],
                            source_type=source["source_type"],
                            published_at=source.get("published_at"),
                            excerpt=source.get("excerpt", ""),
                            relevance=source.get("relevance"),
                        )
                        for source in sources
                    ]
                )

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
            .options(selectinload(ChatMessageRecord.sources))
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
