import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from agentiq.api.deps import get_chat_service, get_request_id
from agentiq.schemas import ContinueRequest, QueryRequest
from agentiq.services.chat_service import ChatService

router = APIRouter()


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\n" f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def streaming_response(
    events,
    request_id: str,
) -> StreamingResponse:
    async def generate():
        async for event in events:
            yield sse_event(event["event"], event["data"])

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


@router.post("/")
async def chat(
    body: QueryRequest,
    request: Request,
    request_id: str = Depends(get_request_id),
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    messages = chat_service.build_messages(body.query)

    return streaming_response(
        chat_service.stream_response(
            messages=messages,
            request_id=request_id,
        ),
        request_id=request_id,
    )


@router.post("/continue")
async def continue_chat(
    body: ContinueRequest,
    request: Request,
    request_id: str = Depends(get_request_id),
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    messages = chat_service.build_continuation_messages(
        query=body.query,
        partial_assistant_content=body.partial_assistant_content,
    )

    return streaming_response(
        chat_service.stream_response(
            messages=messages,
            request_id=request_id,
        ),
        request_id=request_id,
    )
