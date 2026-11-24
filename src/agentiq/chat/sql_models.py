from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Float,
    String,
    Text,
    Uuid,
    JSON,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from agentiq.learning.sql_models import Base


class ConversationRecord(Base):
    """SQL persistence model for a chat conversation."""

    __tablename__ = "chat_conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    messages: Mapped[list["ChatMessageRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ChatMessageRecord.created_at",
    )


class ChatMessageRecord(Base):
    """SQL persistence model for a chat message."""

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        primary_key=True,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("chat_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text(),
        nullable=False,
    )
    truncated: Mapped[bool] = mapped_column(
        Boolean(),
        nullable=False,
        default=False,
    )
    cited_source_ids: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    invalid_source_ids: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    grounded: Mapped[bool | None] = mapped_column(
        Boolean(),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    conversation: Mapped[ConversationRecord] = relationship(
        back_populates="messages",
    )
    sources: Mapped[list["ChatMessageSourceRecord"]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="ChatMessageSourceRecord.source_number",
    )


class ChatMessageSourceRecord(Base):
    """SQL persistence model for a source cited by a chat message."""

    __tablename__ = "chat_message_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        primary_key=True,
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_number: Mapped[int] = mapped_column(
        nullable=False,
    )
    source_id: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        Text(),
        nullable=False,
    )
    url: Mapped[str] = mapped_column(
        Text(),
        nullable=False,
    )
    source_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    excerpt: Mapped[str] = mapped_column(
        Text(),
        nullable=False,
        default="",
    )
    relevance: Mapped[float | None] = mapped_column(
        Float(),
        nullable=True,
    )

    message: Mapped[ChatMessageRecord] = relationship(
        back_populates="sources",
    )
