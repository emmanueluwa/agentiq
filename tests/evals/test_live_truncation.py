from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agentiq.api.deps import get_chat_service
from agentiq.config import Settings, get_settings
from agentiq.main import app
from agentiq.providers.grok import GrokProvider
from agentiq.services.chat_service import ChatService

pytestmark = pytest.mark.live_eval


@pytest_asyncio.fixture
async def low_token_client() -> AsyncGenerator[AsyncClient, None]:
    settings = get_settings()
    constrained_settings = Settings(
        grok_api_key=settings.grok_api_key,
        grok_base_url=settings.grok_base_url,
        grok_model=settings.grok_model,
        max_completion_tokens=120,
        max_retries=settings.max_retries,
        frontend_origin=settings.frontend_origin,
    )

    chat_service = ChatService(
        provider=GrokProvider(constrained_settings),
        settings=constrained_settings,
    )

    async def get_constrained_chat_service() -> ChatService:
        return chat_service

    app.dependency_overrides[get_chat_service] = get_constrained_chat_service

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        yield client

    app.dependency_overrides.clear()


def parse_sse_events(raw: str) -> list[dict]:
    events = []

    for block in raw.strip().split("\n\n"):
        event_name = None
        data_lines = []

        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())

        if event_name and data_lines:
            events.append(
                {
                    "event": event_name,
                    "data": json.loads("\n".join(data_lines)),
                }
            )

    return events


@pytest.mark.asyncio
async def test_long_response_is_truncated_and_can_be_continued(
    low_token_client: AsyncClient,
) -> None:
    initial_response = await low_token_client.post(
        "/",
        json={
            "query": (
                "Write a comprehensive production guide to building RAG "
                "systems, including retrieval, evaluation, observability, "
                "cost control, and go-to-market strategy."
            )
        },
    )

    assert initial_response.status_code == 200

    events = parse_sse_events(initial_response.text)
    done_event = next(event for event in events if event["event"] == "done")
    done_data = done_event["data"]

    assert done_data["finish_reason"] == "length"
    assert done_data["truncated"] is True

    partial_content = done_data["partial_assistant_content"]
    assert partial_content
    assert len(partial_content.split()) > 20

    continuation_response = await low_token_client.post(
        "/continue",
        json={
            "query": (
                "Write a comprehensive production guide to building RAG "
                "systems, including retrieval, evaluation, observability, "
                "cost control, and go-to-market strategy."
            ),
            "partial_assistant_content": partial_content,
        },
    )

    assert continuation_response.status_code == 200

    continuation_events = parse_sse_events(continuation_response.text)
    continuation_done = next(
        event for event in continuation_events if event["event"] == "done"
    )

    assert continuation_done["data"]["finish_reason"] in {"stop", "length"}
