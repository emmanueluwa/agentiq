from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agentiq.api.deps import get_chat_service
from agentiq.main import app
from agentiq.services.chat_service import ChatService


@dataclass(frozen=True)
class FakeStreamEvent:
    content: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None


class FakeChatService:
    def __init__(self, events: list[FakeStreamEvent]):
        self._events = events
        self.received_messages: list[list[dict]] = []

    def build_messages(self, query: str) -> list[dict]:
        return [
            {"role": "system", "content": "test-system"},
            {"role": "user", "content": query},
        ]

    def build_continuation_messages(
        self,
        query: str,
        partial_assistant_content: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": "test-system"},
            {"role": "user", "content": query},
            {"role": "assistant", "content": partial_assistant_content},
            {
                "role": "user",
                "content": "Continue exactly where you stopped.",
            },
        ]

    async def stream_response(
        self,
        messages: list[dict],
        request_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        self.received_messages.append(messages)

        yield {
            "event": "status",
            "data": {"state": "generating", "request_id": request_id},
        }

        for event in self._events:
            if event.content is not None:
                yield {
                    "event": "token",
                    "data": {"content": event.content},
                }

        final_content = "".join(
            event.content for event in self._events if event.content is not None
        )
        final_reason = next(
            (
                event.finish_reason
                for event in self._events
                if event.finish_reason is not None
            ),
            None,
        )
        truncated = final_reason == "length"

        yield {
            "event": "done",
            "data": {
                "request_id": request_id,
                "finish_reason": final_reason,
                "truncated": truncated,
                "usage": {"completion_tokens": 12, "prompt_tokens": 34},
                "partial_assistant_content": (final_content if truncated else None),
            },
        }


def parse_sse_events(raw: str) -> list[dict]:
    events: list[dict] = []

    for block in raw.strip().split("\n\n"):
        event_name: str | None = None
        data_lines: list[str] = []

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


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as async_client:
        yield async_client


@pytest.fixture
def override_chat_service():
    def _override(service: FakeChatService) -> None:
        async def get_fake_chat_service() -> ChatService:
            return service  # type: ignore[return-value]

        app.dependency_overrides[get_chat_service] = get_fake_chat_service

    yield _override

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_truncated_stream_emits_production_sse_contract(
    client: AsyncClient,
    override_chat_service,
) -> None:
    fake_service = FakeChatService(
        events=[
            FakeStreamEvent(content="Partial answer "),
            FakeStreamEvent(content="that stops abruptly."),
            FakeStreamEvent(finish_reason="length"),
        ]
    )

    override_chat_service(fake_service)

    response = await client.post(
        "/",
        json={"query": "Explain distributed tracing."},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-request-id"]

    events = parse_sse_events(response.text)

    assert [event["event"] for event in events] == [
        "status",
        "token",
        "token",
        "done",
    ]

    status_event = events[0]
    assert status_event["data"]["state"] == "generating"
    assert status_event["data"]["request_id"]

    token_events = events[1:-1]
    streamed_content = "".join(event["data"]["content"] for event in token_events)
    assert streamed_content == "Partial answer that stops abruptly."

    done_event = events[-1]
    done_data = done_event["data"]

    assert done_data["finish_reason"] == "length"
    assert done_data["truncated"] is True
    assert done_data["partial_assistant_content"] == (
        "Partial answer that stops abruptly."
    )
    assert done_data["usage"] == {
        "completion_tokens": 12,
        "prompt_tokens": 34,
    }
    assert done_data["request_id"] == status_event["data"]["request_id"]


@pytest.mark.asyncio
async def test_continuation_request_passes_partial_content_to_service(
    client: AsyncClient,
    override_chat_service,
) -> None:
    fake_service = FakeChatService(
        events=[
            FakeStreamEvent(content="Continued answer."),
            FakeStreamEvent(finish_reason="stop"),
        ]
    )

    override_chat_service(fake_service)

    response = await client.post(
        "/continue",
        json={
            "query": "Explain distributed tracing.",
            "partial_assistant_content": "Partial answer that stops abruptly.",
        },
    )

    assert response.status_code == 200

    events = parse_sse_events(response.text)
    done_event = events[-1]

    assert done_event["event"] == "done"
    assert done_event["data"]["finish_reason"] == "stop"
    assert done_event["data"]["truncated"] is False
    assert done_event["data"]["partial_assistant_content"] is None

    messages = fake_service.received_messages[0]

    assert messages[2] == {
        "role": "assistant",
        "content": "Partial answer that stops abruptly.",
    }
    assert messages[3]["role"] == "user"
    assert "Continue exactly where you stopped." in messages[3]["content"]


@pytest.mark.asyncio
async def test_non_truncated_response_does_not_return_partial_content(
    client: AsyncClient,
    override_chat_service,
) -> None:
    fake_service = FakeChatService(
        events=[
            FakeStreamEvent(content="Complete answer."),
            FakeStreamEvent(finish_reason="stop"),
        ]
    )

    override_chat_service(fake_service)

    response = await client.post(
        "/",
        json={"query": "Explain distributed tracing."},
    )

    assert response.status_code == 200

    done_event = parse_sse_events(response.text)[-1]

    assert done_event["data"]["finish_reason"] == "stop"
    assert done_event["data"]["truncated"] is False
    assert done_event["data"]["partial_assistant_content"] is None
