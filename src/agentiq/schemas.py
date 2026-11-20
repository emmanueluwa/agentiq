from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    conversation_id: UUID | None = None


class ContinueRequest(BaseModel):
    conversation_id: UUID
    assistant_message_id: UUID


class ChatMessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    truncated: bool
    created_at: str


class ChatConversationResponse(BaseModel):
    id: UUID
    messages: list[ChatMessageResponse]
