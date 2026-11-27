from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

BASE_URL = os.environ.get("AGENTIQ_TEST_BASE_URL", "http://127.0.0.1:8000")

CASES = [
    {
        "id": "grounded-001",
        "query": "What is distributed tracing, according to OpenTelemetry documentation?",
        "expected": "release",
    },
    {
        "id": "grounded-002",
        "query": "What is a span in OpenTelemetry distributed tracing?",
        "expected": "release",
    },
    {
        "id": "grounded-003",
        "query": "How does context propagation work in OpenTelemetry distributed tracing?",
        "expected": "release",
    },
    {
        "id": "insufficient-001",
        "query": "What is AgentIQ’s internal ARR for this quarter?",
        "expected": "insufficient_evidence",
    },
    {
        "id": "insufficient-002",
        "query": "What will be the exact price of AgentIQ in 2030?",
        "expected": "insufficient_evidence",
    },
]


def parse_sse_events(response_text: str) -> list[dict[str, Any]]:
    import json

    events: list[dict[str, Any]] = []
    event_name: str | None = None

    for line in response_text.splitlines():
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
            continue

        if line.startswith("data:") and event_name:
            payload = json.loads(line.removeprefix("data:").strip())
            events.append({"event": event_name, "data": payload})
            event_name = None

    return events


def extract_done_event(response_text: str) -> dict[str, Any]:
    events = parse_sse_events(response_text)

    for event in events:
        if event["event"] == "done":
            return event

    raise AssertionError("No done event was returned.")


def extract_assistant_response(response_text: str) -> str:
    events = parse_sse_events(response_text)

    return "".join(
        event["data"]["content"] for event in events if event["event"] == "token"
    )


@pytest.mark.live_eval
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=lambda case: case["id"],
)
async def test_research_grounding(case: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{BASE_URL}/",
            json={"query": case["query"]},
        )

    assert response.status_code == 200

    done = extract_done_event(response.text)
    assistant_response = extract_assistant_response(response.text)

    if case["expected"] == "release":
        assert assistant_response.strip(), f"{case['id']} returned no answer"
        assert done["data"]["finish_reason"] in {"stop", "length"}
    else:
        normalized = assistant_response.lower()

        assert (
            "don’t have enough" in normalized
            or "can’t provide a verified answer" in normalized
        )
        assert done["data"]["finish_reason"] == "stop"
