from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for learning-domain SQL models."""


class LearnerProfileRecord(Base):
    """SQL persistence model for a learner profile."""

    __tablename__ = "learner_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        primary_key=True,
    )
    version: Mapped[int] = mapped_column(
        Integer(),
        nullable=False,
        default=1,
    )
    experience_level: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    primary_goal: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    profile_data: Mapped[dict] = mapped_column(
        Text(),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    evidence: Mapped[list["AssessmentEvidenceRecord"]] = relationship(
        back_populates="profile",
        cascade="all, delete-orphan",
    )


class AssessmentEvidenceRecord(Base):
    """SQL persistence model for assessment evidence."""

    __tablename__ = "assessment_evidence"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        primary_key=True,
    )
    profile_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("learner_profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    assessment_result_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        nullable=False,
        unique=True,
    )
    topic: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    evidence_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    score: Mapped[Decimal] = mapped_column(
        Numeric(5, 2),
        nullable=False,
    )
    passed: Mapped[bool] = mapped_column(
        Boolean(),
        nullable=False,
    )
    evidence_data: Mapped[dict] = mapped_column(
        Text(),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    profile: Mapped[LearnerProfileRecord] = relationship(
        back_populates="evidence",
    )
