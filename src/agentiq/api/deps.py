from __future__ import annotations

import uuid
from collections.abc import Callable

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from agentiq.chat.unit_of_work import (
    SqlChatUnitOfWork,
    create_chat_unit_of_work_factory,
)
from agentiq.plans.unit_of_work import (
    SqlPlanUnitOfWork,
    create_plan_unit_of_work_factory,
)
from agentiq.config import Settings, get_settings
from agentiq.providers.grok import GrokProvider
from agentiq.services.chat_service import ChatService


def get_chat_service(
    settings: Settings = Depends(get_settings),
) -> ChatService:
    if not settings.grok_api_key:
        raise HTTPException(
            status_code=500,
            detail="GROK_API_KEY environment variable is missing.",
        )

    return ChatService(
        provider=GrokProvider(settings),
        settings=settings,
    )


def get_request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def get_chat_unit_of_work_factory(
    request: Request,
) -> Callable[[], SqlChatUnitOfWork]:
    session_factory = request.app.state.session_factory

    return create_chat_unit_of_work_factory(session_factory)


def get_plan_unit_of_work_factory(
    request: Request,
) -> Callable[[], SqlPlanUnitOfWork]:
    session_factory = request.app.state.session_factory

    return create_plan_unit_of_work_factory(session_factory)
