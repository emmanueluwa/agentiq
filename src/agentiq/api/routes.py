from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from agentiq.api.deps import (
    get_chat_service,
    get_chat_unit_of_work_factory,
    get_request_id,
)
from agentiq.chat.repository import ConversationNotFoundError
from agentiq.chat.unit_of_work import SqlChatUnitOfWork
from agentiq.schemas import (
    ChatConversationResponse,
    ChatMessageResponse,
    ContinueRequest,
    QueryRequest,
)
from agentiq.services.chat_service import ChatService

router = APIRouter()


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\n" f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def streaming_response(
    events: AsyncGenerator[dict, None], request_id: str
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
    chat_unit_of_work_factory: Callable[[], SqlChatUnitOfWork] = Depends(
        get_chat_unit_of_work_factory,
    ),
) -> StreamingResponse:
    async with chat_unit_of_work_factory() as unit_of_work:
        if body.conversation_id:
            try:
                await unit_of_work.get_conversation(body.conversation_id)
            except ConversationNotFoundError as error:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(error),
                ) from error

            conversation_id = body.conversation_id
        else:
            conversation = await unit_of_work.create_conversation()
            conversation_id = conversation.id

        await unit_of_work.append_message(
            conversation_id=conversation_id,
            role="user",
            content=body.query,
        )

        assistant_message = await unit_of_work.append_message(
            conversation_id=conversation_id,
            role="assistant",
            content="",
        )

    assistant_message_id = assistant_message.id
    messages = chat_service.build_messages(body.query)

    async def generate() -> AsyncGenerator[dict, None]:
        content_parts: list[str] = []
        truncated = False

        try:
            async for event in chat_service.stream_response(
                messages=messages,
                request_id=request_id,
                conversation_id=str(conversation_id),
                assistant_message_id=str(assistant_message_id),
            ):
                if event["event"] == "token":
                    content_parts.append(event["data"]["content"])

                if event["event"] == "done":
                    truncated = event["data"]["truncated"]

                yield event
        finally:
            assistant_content = "".join(content_parts)

            if assistant_content:
                async with chat_unit_of_work_factory() as persistence_unit_of_work:
                    await persistence_unit_of_work.update_message(
                        message_id=assistant_message_id,
                        content=assistant_content,
                        truncated=truncated,
                    )

    return streaming_response(generate(), request_id)


@router.post("/continue")
async def continue_chat(
    body: ContinueRequest,
    request: Request,
    request_id: str = Depends(get_request_id),
    chat_service: ChatService = Depends(get_chat_service),
    chat_unit_of_work_factory: Callable[[], SqlChatUnitOfWork] = Depends(
        get_chat_unit_of_work_factory,
    ),
) -> StreamingResponse:
    async with chat_unit_of_work_factory() as unit_of_work:
        try:
            await unit_of_work.get_conversation(body.conversation_id)
            assistant_message = await unit_of_work.get_message(
                body.assistant_message_id
            )
        except ConversationNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        if assistant_message.conversation_id != body.conversation_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message does not belong to the conversation.",
            )

        conversation_messages = await unit_of_work.get_messages(
            body.conversation_id,
        )

        assistant_message_index = next(
            (
                index
                for index, message in enumerate(conversation_messages)
                if message.id == assistant_message.id
            ),
            None,
        )

        if assistant_message_index is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message does not belong to the conversation.",
            )

        preceding_user_message = next(
            (
                message
                for message in reversed(conversation_messages[:assistant_message_index])
                if message.role == "user"
            ),
            None,
        )

        if preceding_user_message is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message precedes the truncated assistant message.",
            )

        messages = chat_service.build_continuation_messages(
            query=preceding_user_message.content,
            partial_assistant_content=assistant_message.content,
        )

        continuation_message = await unit_of_work.append_message(
            conversation_id=body.conversation_id,
            role="assistant",
            content="",
        )

    continuation_message_id = continuation_message.id

    async def generate() -> AsyncGenerator[dict, None]:
        content_parts: list[str] = []
        truncated = False

        try:
            async for event in chat_service.stream_response(
                messages=messages,
                request_id=request_id,
                conversation_id=str(body.conversation_id),
                assistant_message_id=str(continuation_message_id),
            ):
                if event["event"] == "token":
                    content_parts.append(event["data"]["content"])

                if event["event"] == "done":
                    truncated = event["data"]["truncated"]

                yield event
        finally:
            assistant_content = "".join(content_parts)

            if assistant_content:
                async with chat_unit_of_work_factory() as persistence_unit_of_work:
                    await persistence_unit_of_work.update_message(
                        message_id=continuation_message_id,
                        content=assistant_content,
                        truncated=truncated,
                    )

    return streaming_response(generate(), request_id)


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    chat_unit_of_work_factory: Callable[[], SqlChatUnitOfWork] = Depends(
        get_chat_unit_of_work_factory,
    ),
) -> ChatConversationResponse:
    async with chat_unit_of_work_factory() as unit_of_work:
        try:
            messages = await unit_of_work.get_messages(conversation_id)
        except ConversationNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    return ChatConversationResponse(
        id=conversation_id,
        messages=[
            ChatMessageResponse(
                id=message.id,
                role=message.role,
                content=message.content,
                truncated=message.truncated,
                created_at=message.created_at.isoformat(),
            )
            for message in messages
        ],
    )
