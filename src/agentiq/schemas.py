from uuid import UUID
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    conversation_id: UUID | None = None


class ContinueRequest(BaseModel):
    conversation_id: UUID
    assistant_message_id: UUID


class ResearchSourceResponse(BaseModel):
    id: str
    source_number: int
    title: str
    url: str
    source_type: str
    published_at: str | None
    excerpt: str
    relevance: float | None


class ChatMessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    truncated: bool
    created_at: str
    cited_source_ids: list[str] = []
    invalid_source_ids: list[str] = []
    grounded: bool | None = None
    sources: list[ResearchSourceResponse] = []


class ChatConversationResponse(BaseModel):
    id: UUID
    messages: list[ChatMessageResponse]


class PlanItemCreate(BaseModel):
    title: str
    details: str = ""
    due_at: datetime | None = None


class PlanCreate(BaseModel):
    conversation_id: UUID
    source_message_id: UUID
    title: str
    outcome: str
    items: list[PlanItemCreate]


class PlanItemUpdate(BaseModel):
    title: str | None = None
    details: str | None = None
    status: Literal["todo", "in_progress", "done", "skipped"] | None = None
    due_at: datetime | None = None


class PlanItemResponse(BaseModel):
    id: UUID
    title: str
    details: str
    status: str
    position: int
    due_at: datetime | None


class PlanResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    source_message_id: UUID
    title: str
    outcome: str
    status: str
    items: list[PlanItemResponse]


class PlanDraftRequest(BaseModel):
    conversation_id: UUID
    source_message_id: UUID


class PlanDraftItem(BaseModel):
    title: str
    details: str = ""


class PlanDraftResponse(BaseModel):
    title: str
    outcome: str
    items: list[PlanDraftItem]
