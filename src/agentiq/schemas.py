from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)


class ContinueRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    partial_assistant_content: str = Field(..., min_length=1)
