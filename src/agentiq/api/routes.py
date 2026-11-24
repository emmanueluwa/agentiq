from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from agentiq.api.deps import (
    get_chat_service,
    get_chat_unit_of_work_factory,
    get_plan_unit_of_work_factory,
    get_request_id,
)
from agentiq.chat.repository import ConversationNotFoundError
from agentiq.chat.unit_of_work import SqlChatUnitOfWork
from agentiq.plans.repository import (
    PlanItemNotFoundError,
    PlanNotFoundError,
)
from agentiq.plans.unit_of_work import SqlPlanUnitOfWork
from agentiq.research.delivery import deliver_grounded_answer
from agentiq.research.dependencies import get_research_service
from agentiq.research.models import (
    ResearchEvidence,
    ResearchPreparationError,
    SourceCandidate,
)
from agentiq.research.service import ResearchService
from agentiq.schemas import (
    ChatConversationResponse,
    ChatMessageResponse,
    ContinueRequest,
    PlanCreate,
    PlanDraftRequest,
    PlanDraftResponse,
    PlanItemResponse,
    PlanItemUpdate,
    PlanResponse,
    QueryRequest,
    ResearchSourceResponse,
)
from agentiq.services.chat_service import ChatService, PlanDraftGenerationError

router = APIRouter()


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\n" f"data: {json.dumps(data, separators=(',', ':'))}\n\n"


def streaming_response(
    events: AsyncGenerator[dict, None],
    request_id: str,
) -> StreamingResponse:
    async def generate() -> AsyncGenerator[str, None]:
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
    research_service: ResearchService = Depends(get_research_service),
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

    try:
        research = await research_service.prepare(body.query)
    except ResearchPreparationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    if research.insufficient_evidence:
        assistant_content = research_service.insufficient_evidence_response(
            body.query,
        )

        async with chat_unit_of_work_factory() as persistence_unit_of_work:
            await persistence_unit_of_work.update_message(
                message_id=assistant_message_id,
                content=assistant_content,
                truncated=False,
                cited_source_ids=[],
                invalid_source_ids=[],
                grounded=False,
                sources=[],
            )

        async def insufficient_evidence_stream() -> AsyncGenerator[dict, None]:
            yield {
                "event": "status",
                "data": {
                    "state": "insufficient_evidence",
                    "request_id": request_id,
                    "conversation_id": str(conversation_id),
                    "assistant_message_id": str(assistant_message_id),
                },
            }

            yield {
                "event": "grounding",
                "data": {
                    "cited_source_ids": [],
                    "invalid_source_ids": [],
                    "grounded": False,
                    "sources": [],
                },
            }

            yield {
                "event": "token",
                "data": {"content": assistant_content},
            }

            yield {
                "event": "done",
                "data": {
                    "request_id": request_id,
                    "conversation_id": str(conversation_id),
                    "assistant_message_id": str(assistant_message_id),
                    "finish_reason": "stop",
                    "truncated": False,
                    "usage": None,
                    "partial_assistant_content": None,
                },
            }

        return streaming_response(
            insufficient_evidence_stream(),
            request_id,
        )

    if research.messages is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research preparation did not produce generation messages.",
        )

    return streaming_response(
        deliver_grounded_answer(
            messages=research.messages,
            evidence=research.evidence,
            chat_service=chat_service,
            research_service=research_service,
            unit_of_work_factory=chat_unit_of_work_factory,
            request_id=request_id,
            conversation_id=conversation_id,
            assistant_message_id=assistant_message_id,
        ),
        request_id,
    )


@router.post("/continue")
async def continue_chat(
    body: ContinueRequest,
    request: Request,
    request_id: str = Depends(get_request_id),
    chat_service: ChatService = Depends(get_chat_service),
    research_service: ResearchService = Depends(get_research_service),
    chat_unit_of_work_factory: Callable[[], SqlChatUnitOfWork] = Depends(
        get_chat_unit_of_work_factory,
    ),
) -> StreamingResponse:
    async with chat_unit_of_work_factory() as unit_of_work:
        try:
            await unit_of_work.get_conversation(body.conversation_id)

            requested_message = await unit_of_work.get_message(
                body.assistant_message_id,
            )
        except ConversationNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        if requested_message.conversation_id != body.conversation_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message does not belong to the conversation.",
            )

        if requested_message.role != "assistant" or not requested_message.truncated:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only truncated assistant messages can be continued.",
            )

        # get_messages() eagerly loads sources in the repository.
        conversation_messages = await unit_of_work.get_messages(
            body.conversation_id,
        )

        assistant_message_index = next(
            (
                index
                for index, message in enumerate(conversation_messages)
                if message.id == body.assistant_message_id
            ),
            None,
        )

        if assistant_message_index is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Message does not belong to the conversation.",
            )

        assistant_message = conversation_messages[assistant_message_index]

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

        evidence = ResearchEvidence(
            query=preceding_user_message.content,
            sources=[
                SourceCandidate(
                    id=source.source_id,
                    title=source.title,
                    url=source.url,
                    source_type=source.source_type,
                    published_at=source.published_at,
                    excerpt=source.excerpt or "",
                    relevance=source.relevance,
                )
                for source in assistant_message.sources
            ],
        )

        if not evidence.sources:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "This message has no saved research evidence. "
                    "Submit the question again to generate a cited answer."
                ),
            )

        messages = research_service.build_messages(
            query=evidence.query,
            evidence=evidence,
        )

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                },
                {
                    "role": "user",
                    "content": (
                        "Continue the preceding answer without repeating it. "
                        "Use only the supplied research excerpts for factual "
                        "claims and preserve their original citation IDs. "
                        "Make the continuation understandable on its own. "
                        "Do not start with an unfinished sentence fragment. "
                        "If the excerpts cannot support further claims, "
                        "state that limitation rather than inventing details."
                    ),
                },
            ]
        )

        continuation_message = await unit_of_work.append_message(
            conversation_id=body.conversation_id,
            role="assistant",
            content="",
        )

        continuation_message_id = continuation_message.id

    return streaming_response(
        deliver_grounded_answer(
            messages=messages,
            evidence=evidence,
            chat_service=chat_service,
            research_service=research_service,
            unit_of_work_factory=chat_unit_of_work_factory,
            request_id=request_id,
            conversation_id=body.conversation_id,
            assistant_message_id=continuation_message_id,
        ),
        request_id,
    )


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
                    cited_source_ids=message.cited_source_ids,
                    invalid_source_ids=message.invalid_source_ids,
                    grounded=message.grounded,
                    sources=[
                        ResearchSourceResponse(
                            id=source.source_id,
                            source_number=source.source_number,
                            title=source.title,
                            url=source.url,
                            source_type=source.source_type,
                            published_at=(
                                source.published_at.isoformat()
                                if source.published_at
                                else None
                            ),
                            excerpt=source.excerpt,
                            relevance=source.relevance,
                        )
                        for source in message.sources
                    ],
                )
                for message in messages
            ],
        )


def serialize_plan(plan: object) -> PlanResponse:
    return PlanResponse(
        id=plan.id,
        conversation_id=plan.conversation_id,
        source_message_id=plan.source_message_id,
        title=plan.title,
        outcome=plan.outcome,
        status=plan.status,
        items=[
            PlanItemResponse(
                id=item.id,
                title=item.title,
                details=item.details,
                status=item.status,
                position=item.position,
                due_at=item.due_at,
            )
            for item in plan.items
        ],
    )


@router.post("/plans/drafts")
async def create_plan_draft(
    body: PlanDraftRequest,
    chat_service: ChatService = Depends(get_chat_service),
    chat_unit_of_work_factory: Callable[[], SqlChatUnitOfWork] = Depends(
        get_chat_unit_of_work_factory,
    ),
) -> PlanDraftResponse:
    async with chat_unit_of_work_factory() as chat_unit_of_work:
        try:
            await chat_unit_of_work.get_conversation(body.conversation_id)

            source_message = await chat_unit_of_work.get_message(
                body.source_message_id,
            )
        except ConversationNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        if source_message.conversation_id != body.conversation_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message does not belong to the conversation.",
            )

        source_content = source_message.content

    try:
        draft = await chat_service.create_plan_draft(
            source_message=source_content,
        )
    except PlanDraftGenerationError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error

    return PlanDraftResponse(**draft)


@router.post("/plans", status_code=status.HTTP_201_CREATED)
async def create_plan(
    body: PlanCreate,
    chat_unit_of_work_factory: Callable[[], SqlChatUnitOfWork] = Depends(
        get_chat_unit_of_work_factory,
    ),
    plan_unit_of_work_factory: Callable[[], SqlPlanUnitOfWork] = Depends(
        get_plan_unit_of_work_factory,
    ),
) -> PlanResponse:
    async with chat_unit_of_work_factory() as chat_unit_of_work:
        try:
            await chat_unit_of_work.get_conversation(body.conversation_id)

            source_message = await chat_unit_of_work.get_message(
                body.source_message_id,
            )
        except ConversationNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        if source_message.conversation_id != body.conversation_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message does not belong to the conversation.",
            )

    async with plan_unit_of_work_factory() as plan_unit_of_work:
        plan = await plan_unit_of_work.create_plan(
            conversation_id=body.conversation_id,
            source_message_id=body.source_message_id,
            title=body.title,
            outcome=body.outcome,
            items=[item.model_dump() for item in body.items],
        )

        return serialize_plan(plan)


@router.get("/plans")
async def list_plans(
    plan_unit_of_work_factory: Callable[[], SqlPlanUnitOfWork] = Depends(
        get_plan_unit_of_work_factory,
    ),
) -> list[PlanResponse]:
    async with plan_unit_of_work_factory() as plan_unit_of_work:
        plans = await plan_unit_of_work.list_plans()

        return [serialize_plan(plan) for plan in plans]


@router.get("/plans/{plan_id}")
async def get_plan(
    plan_id: UUID,
    plan_unit_of_work_factory: Callable[[], SqlPlanUnitOfWork] = Depends(
        get_plan_unit_of_work_factory,
    ),
) -> PlanResponse:
    async with plan_unit_of_work_factory() as plan_unit_of_work:
        try:
            plan = await plan_unit_of_work.get_plan(plan_id)
        except PlanNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        return serialize_plan(plan)


@router.patch("/plans/{plan_id}/items/{item_id}")
async def update_plan_item(
    plan_id: UUID,
    item_id: UUID,
    body: PlanItemUpdate,
    plan_unit_of_work_factory: Callable[[], SqlPlanUnitOfWork] = Depends(
        get_plan_unit_of_work_factory,
    ),
) -> PlanItemResponse:
    async with plan_unit_of_work_factory() as plan_unit_of_work:
        try:
            item = await plan_unit_of_work.update_plan_item(
                plan_id=plan_id,
                item_id=item_id,
                status=body.status,
                title=body.title,
                details=body.details,
                due_at=body.due_at,
            )
        except (PlanNotFoundError, PlanItemNotFoundError) as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

        return PlanItemResponse(
            id=item.id,
            title=item.title,
            details=item.details,
            status=item.status,
            position=item.position,
            due_at=item.due_at,
        )


@router.delete("/plans/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_plan(
    plan_id: UUID,
    plan_unit_of_work_factory: Callable[[], SqlPlanUnitOfWork] = Depends(
        get_plan_unit_of_work_factory,
    ),
) -> None:
    async with plan_unit_of_work_factory() as plan_unit_of_work:
        try:
            await plan_unit_of_work.delete_plan(plan_id)
        except PlanNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error
