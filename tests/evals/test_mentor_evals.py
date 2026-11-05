from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agentiq.main import app
from tests.evals.grader import (
    judge_response,
    passes_production_bar,
    write_eval_report,
)

pytestmark = pytest.mark.live_eval

EVAL_FILE = Path(__file__).parent / "mentor_evals.jsonl"
REPORT_RESULTS: list[dict] = []


def load_cases() -> list[dict]:
    return [
        json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()
    ]


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


def extract_assistant_response(raw: str) -> str:
    events = parse_sse_events(raw)

    token_events = [event for event in events if event["event"] == "token"]

    return "".join(event["data"].get("content", "") for event in token_events)


def extract_done_event(raw: str) -> dict:
    events = parse_sse_events(raw)

    done_events = [event for event in events if event["event"] == "done"]

    if not done_events:
        raise AssertionError("Response did not contain a done event")

    return done_events[-1]["data"]


def assert_deterministic_criteria(
    case: dict,
    assistant_response: str,
    done_data: dict,
) -> None:
    body = assistant_response.lower()

    for phrase in case["must_include"]:
        assert phrase.lower() in body, f"{case['id']} missing required phrase: {phrase}"

    for phrase in case["must_not_include"]:
        assert (
            phrase.lower() not in body
        ), f"{case['id']} contains forbidden phrase: {phrase}"

    if case["requires_citation"]:
        citation_markers = [
            "source",
            "according to",
            "reference",
            "documentation",
            "best practice",
        ]

        assert any(
            marker in body for marker in citation_markers
        ), f"{case['id']} must cite or identify a source"

    if case["must_admit_unknown"]:
        uncertainty_markers = [
            "don't know",
            "do not know",
            "cannot verify",
            "can't verify",
            "not available",
            "no public information",
            "i can't",
            "i cannot",
        ]

        assert any(
            marker in body for marker in uncertainty_markers
        ), f"{case['id']} must clearly express uncertainty"

    if case["expected_length"] == "short":
        assert (
            len(assistant_response.split()) <= 350
        ), f"{case['id']} expected a short response"

    if case["expected_length"] == "medium":
        assert (
            len(assistant_response.split()) >= 50
        ), f"{case['id']} expected a substantive response"

    if case["category"] == "truncation":
        assert done_data["truncated"] is True, f"{case['id']} expected truncation"
        assert done_data["finish_reason"] == "length"
        assert done_data[
            "partial_assistant_content"
        ], f"{case['id']} expected preserved partial content"


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    load_cases(),
    ids=lambda case: case["id"],
)
async def test_mentor_live_eval(case: dict, client: AsyncClient):
    response = await client.post(
        "/",
        json={"query": case["input"]},
    )

    assert response.status_code == 200

    done_data = extract_done_event(response.text)
    assistant_response = extract_assistant_response(response.text)

    assert (
        assistant_response.strip()
    ), f"{case['id']} returned an empty assistant response"

    assert_deterministic_criteria(
        case=case,
        assistant_response=assistant_response,
        done_data=done_data,
    )

    judge = await judge_response(
        question=case["input"],
        response=assistant_response,
    )

    passed = passes_production_bar(judge)

    REPORT_RESULTS.append(
        {
            "id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "response": assistant_response,
            "done": done_data,
            "judge": judge.__dict__,
            "passed": passed,
        }
    )

    assert passed, (
        f"{case['id']} failed the production quality bar.\n"
        f"Judge result: {judge}\n"
        f"Response:\n{assistant_response}"
    )


def pytest_sessionfinish(session, exitstatus):
    if REPORT_RESULTS:
        write_eval_report(REPORT_RESULTS)
