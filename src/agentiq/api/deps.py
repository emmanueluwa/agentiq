import uuid


from fastapi import Depends, HTTPException, Request

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
