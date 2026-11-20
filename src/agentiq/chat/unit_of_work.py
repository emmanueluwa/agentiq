from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from agentiq.chat.repository import ChatRepository
from agentiq.chat.sql_models import ChatMessageRecord, ConversationRecord


class SqlChatUnitOfWork:
    """Transactional unit of work for chat persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repository = ChatRepository(session)

    async def create_conversation(self) -> ConversationRecord:
        return await self._repository.create_conversation()

    async def get_conversation(self, conversation_id: UUID) -> ConversationRecord:
        return await self._repository.get_conversation(conversation_id)

    async def append_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        *,
        message_id: UUID | None = None,
        truncated: bool = False,
    ) -> ChatMessageRecord:
        return await self._repository.append_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            message_id=message_id,
            truncated=truncated,
        )

    async def update_message(
        self,
        message_id: UUID,
        *,
        content: str,
        truncated: bool,
    ) -> ChatMessageRecord:
        return await self._repository.update_message(
            message_id=message_id,
            content=content,
            truncated=truncated,
        )

    async def get_messages(self, conversation_id: UUID) -> list[ChatMessageRecord]:
        return await self._repository.get_messages(conversation_id)

    async def get_message(self, message_id: UUID) -> ChatMessageRecord:
        return await self._repository.get_message(message_id)

    async def __aenter__(self) -> SqlChatUnitOfWork:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        if exc_type is not None:
            await self._session.rollback()
            return

        await self._session.commit()


def create_chat_unit_of_work_factory(
    session_factory: Callable[[], AsyncSession],
) -> Callable[[], SqlChatUnitOfWork]:
    """Create a factory that opens one transaction per chat operation."""

    def create_unit_of_work() -> SqlChatUnitOfWork:
        return SqlChatUnitOfWork(session_factory())

    return create_unit_of_work
