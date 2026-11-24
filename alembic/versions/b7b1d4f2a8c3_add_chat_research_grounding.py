"""add chat research grounding

Revision ID: b7b1d4f2a8c3
Revises: da4656e6d5c6
Create Date: 2026-11-23

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "b7b1d4f2a8c3"
down_revision: Union[str, Sequence[str], None] = "da4656e6d5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column(
            "cited_source_ids",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "chat_messages",
        sa.Column(
            "invalid_source_ids",
            JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "chat_messages",
        sa.Column("grounded", sa.Boolean(), nullable=True),
    )

    op.create_table(
        "chat_message_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("source_number", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.String(length=32), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("excerpt", sa.Text(), nullable=False),
        sa.Column("relevance", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["chat_messages.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "message_id",
            "source_number",
            name="uq_chat_message_sources_message_source_number",
        ),
    )

    op.create_index(
        "ix_chat_message_sources_message_id",
        "chat_message_sources",
        ["message_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_chat_message_sources_message_id",
        table_name="chat_message_sources",
    )
    op.drop_table("chat_message_sources")
    op.drop_column("chat_messages", "grounded")
    op.drop_column("chat_messages", "invalid_source_ids")
    op.drop_column("chat_messages", "cited_source_ids")
