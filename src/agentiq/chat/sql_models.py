from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    conversation: Mapped[ConversationRecord] = relationship(
        back_populates="messages",
    )
