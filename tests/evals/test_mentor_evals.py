from __future__ import annotations

from dataclasses import asdict
import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agentiq.main import app
from tests.evals.grader import (
    has_private_value_leakage,
    judge_quality,
    judge_safety,
    passes_quality_bar,
    passes_safety_bar,
    write_eval_report,
)

pytestmark = pytest.mark.live_eval

EVAL_FILE = Path(__file__).parent / "mentor_evals.jsonl"
CHAT_ENDPOINT = "/"
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
    return "".join(
        event["data"].get("content", "")
        for event in parse_sse_events(raw)
        if event["event"] == "token"
    )


def extract_done_event(raw: str) -> dict:
    done_events = [event for event in parse_sse_events(raw) if event["event"] == "done"]

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

    if case["safety_policy"] == "privacy":
        assert not has_private_value_leakage(
            assistant_response
        ), f"{case['id']} appears to disclose private AgentIQ information"

    if case["expected_length"] == "short":
        assert (
            len(assistant_response.split()) <= 350
        ), f"{case['id']} expected a short response"

    if case["expected_length"] == "medium":
        assert (
            len(assistant_response.split()) >= 50
        ), f"{case['id']} expected a substantive response"


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
        CHAT_ENDPOINT,
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

    if case["safety_policy"] == "privacy":
        safety = await judge_safety(
            question=case["input"],
            response=assistant_response,
        )

        passed = passes_safety_bar(safety)

        REPORT_RESULTS.append(
            {
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "response": assistant_response,
                "done": done_data,
                "safety": asdict(safety),
                "passed": passed,
            }
        )

        assert passed, (
            f"{case['id']} failed the privacy and safety policy bar.\n"
            f"Safety result: {safety}\n"
            f"Response:\n{assistant_response}"
        )

        return

    quality = await judge_quality(
        question=case["input"],
        response=assistant_response,
    )

    passed = passes_quality_bar(quality, case["category"])

    REPORT_RESULTS.append(
        {
            "id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "response": assistant_response,
            "done": done_data,
            "judge": asdict(quality),
            "passed": passed,
        }
    )

    assert passed, (
        f"{case['id']} failed the production quality bar.\n"
        f"Judge result: {quality}\n"
        f"Response:\n{assistant_response}"
    )


def pytest_sessionfinish(session, exitstatus):
    if REPORT_RESULTS:
        write_eval_report(REPORT_RESULTS)
