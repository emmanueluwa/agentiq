from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agentiq.learning.sql_models import Base


class PlanRecord(Base):
    """SQL persistence model for a user-reviewed plan."""

    __tablename__ = "plans"

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
    source_message_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("chat_messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    outcome: Mapped[str] = mapped_column(
        Text(),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    items: Mapped[list["PlanItemRecord"]] = relationship(
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="PlanItemRecord.position",
    )


class PlanItemRecord(Base):
    """SQL persistence model for an action item within a plan."""

    __tablename__ = "plan_items"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        primary_key=True,
    )
    plan_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    details: Mapped[str] = mapped_column(
        Text(),
        nullable=False,
        default="",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="todo",
    )
    position: Mapped[int] = mapped_column(
        Integer(),
        nullable=False,
    )
    due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    plan: Mapped[PlanRecord] = relationship(
        back_populates="items",
    )
