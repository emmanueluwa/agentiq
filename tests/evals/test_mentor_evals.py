import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from agentiq.main import app

pytestmark = pytest.mark.live_eval

EVAL_FILE = Path(__file__).parent / "mentor_evals.jsonl"


def load_cases() -> list[dict]:
    return [
        json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()
    ]


@pytest.fixture
async def client():
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
async def test_mentor_eval(case: dict, client: AsyncClient):
    response = await client.post(
        "/",  # change if your real endpoint differs
        json={"query": case["input"]},
    )

    assert response.status_code == 200

    body = response.text.lower()

    for phrase in case["must_include"]:
        assert phrase.lower() in body, f"{case['id']} missing required phrase: {phrase}"

    for phrase in case["must_not_include"]:
        assert (
            phrase.lower() not in body
        ), f"{case['id']} contains forbidden phrase: {phrase}"

    if case["requires_citation"]:
        assert "source" in body, f"{case['id']} must cite a source"

    if case["must_admit_unknown"]:
        uncertainty_markers = [
            "don't know",
            "do not know",
            "cannot verify",
            "not available",
            "no public information",
        ]

        assert any(
            marker in body for marker in uncertainty_markers
        ), f"{case['id']} must clearly express uncertainty"
