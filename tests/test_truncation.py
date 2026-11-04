import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from agentiq.main import app


def make_chunk(content: str | None, finish_reason: str | None = None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration

        return self._chunks.pop(0)


@pytest.mark.asyncio
async def test_truncated_response_emits_done_with_truncated_true():
    fake_stream = FakeAsyncStream(
        [
            make_chunk("Partial answer "),
            make_chunk("that stops abruptly."),
            make_chunk(None, finish_reason="length"),
        ]
    )

    with patch(
        "agentiq.main.client.chat.completions.create",
        new=AsyncMock(return_value=fake_stream),
    ):
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/",
                json={"query": "Explain distributed tracing."},
            )

    assert response.status_code == 200

    body = response.text

    assert "event: token" in body
    assert "Partial answer" in body
    assert '"truncated":true' in body
    assert '"finish_reason":"length"' in body
    assert "partial_assistant_content" in body


def parse_sse_events(raw: str) -> list[dict]:
    events = []

    for block in raw.strip().split("\n\n"):
        event_name = None
        data = ""

        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()

            if line.startswith("data:"):
                data += line.split(":", 1)[1].strip()

        if event_name and data:
            events.append(
                {
                    "event": event_name,
                    "data": json.loads(data),
                }
            )

    return events


@pytest.mark.asyncio
async def test_truncation_contract_is_valid():
    fake_stream = FakeAsyncStream(
        [
            make_chunk("Partial "),
            make_chunk(None, finish_reason="length"),
        ]
    )

    with patch(
        "agentiq.main.client.chat.completions.create",
        new=AsyncMock(return_value=fake_stream),
    ):
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/",
                json={"query": "Explain distributed tracing."},
            )

    events = parse_sse_events(response.text)
    done_event = next(event for event in events if event["event"] == "done")

    assert done_event["data"]["truncated"] is True
    assert done_event["data"]["finish_reason"] == "length"
    assert done_event["data"]["partial_assistant_content"] == "Partial "
