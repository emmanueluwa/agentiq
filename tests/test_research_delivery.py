import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agentiq.research.delivery import (
    UNVERIFIED_ANSWER_RESPONSE,
    deliver_grounded_answer,
)
from agentiq.research.models import (
    CitationValidation,
    ResearchEvidence,
    SourceCandidate,
)
from agentiq.research.verification import VerificationError

ANSWER = "Rust is a systems programming language. [1]"


class FakeChatService:
    def __init__(self, *, accepted=True, verification_error=False):
        self.accepted = accepted
        self.verification_error = verification_error

    async def stream_response(self, **kwargs):
        yield {"event": "token", "data": {"content": ANSWER}}
        yield {
            "event": "done",
            "data": {
                "finish_reason": "stop",
                "truncated": False,
                "usage": None,
            },
        }

    async def verify_evidence(self, *, content, evidence):
        if self.verification_error:
            raise VerificationError("Verification failed.")

        if self.accepted:
            return SimpleNamespace(
                accepted=True,
                evidence_sufficient=True,
                reviews=[
                    SimpleNamespace(
                        line_id=0,
                        verdict="supported",
                        explanation="Supported by the supplied evidence.",
                        supporting_quotes=[
                            SimpleNamespace(
                                source_id="source:1",
                                quotation="Rust is a systems programming language.",
                            )
                        ],
                    )
                ],
            )

        return SimpleNamespace(
            accepted=False,
            evidence_sufficient=True,
            reviews=[],
        )


class FakeResearchService:
    def validate_citations(self, **kwargs):
        return CitationValidation(
            cited_source_ids=["source:1"],
            invalid_source_ids=[],
        )


class FakeUnitOfWork:
    def __init__(self, *, fail_commit=False):
        self.fail_commit = fail_commit
        self.saved = None
        self.committed = False

    async def __aenter__(self):
        return self

    async def update_message(self, **kwargs):
        self.saved = kwargs

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.fail_commit:
            raise RuntimeError("Test commit failure")

        self.committed = True


def make_stream(chat, unit_of_work):
    return deliver_grounded_answer(
        messages=[],
        evidence=ResearchEvidence(
            query="What kind of language is Rust?",
            sources=[
                SourceCandidate(
                    id="source:1",
                    title="Example",
                    url="https://example.com/rust",
                    excerpt="Rust is a systems programming language.",
                )
            ],
        ),
        chat_service=chat,
        research_service=FakeResearchService(),
        unit_of_work_factory=lambda: unit_of_work,
        request_id="test-request",
        conversation_id=uuid4(),
        assistant_message_id=uuid4(),
    )


async def collect(stream):
    return [event async for event in stream]


def test_answer_is_committed_before_any_answer_token():
    unit_of_work = FakeUnitOfWork()

    async def run():
        events = []

        async for event in make_stream(FakeChatService(), unit_of_work):
            if event["event"] in {"grounding", "token", "done"}:
                assert unit_of_work.committed

            events.append(event)

        return events

    events = asyncio.run(run())

    assert [event["event"] for event in events] == [
        "status",
        "status",
        "grounding",
        "token",
        "done",
    ]
    assert unit_of_work.saved["content"] == ANSWER
    assert unit_of_work.saved["grounded"] is True


def test_rejected_draft_is_replaced_and_never_released():
    unit_of_work = FakeUnitOfWork()

    events = asyncio.run(
        collect(
            make_stream(
                FakeChatService(accepted=False),
                unit_of_work,
            )
        )
    )

    tokens = [event["data"]["content"] for event in events if event["event"] == "token"]

    assert tokens == [UNVERIFIED_ANSWER_RESPONSE]
    assert ANSWER not in tokens
    assert unit_of_work.saved["content"] == UNVERIFIED_ANSWER_RESPONSE
    assert unit_of_work.saved["grounded"] is False
    assert unit_of_work.saved["sources"] == []


@pytest.mark.parametrize("failure", ["verification", "commit"])
def test_operational_failure_releases_no_answer_or_done(failure):
    unit_of_work = FakeUnitOfWork(
        fail_commit=failure == "commit",
    )

    events = asyncio.run(
        collect(
            make_stream(
                FakeChatService(
                    verification_error=failure == "verification",
                ),
                unit_of_work,
            )
        )
    )

    event_names = [event["event"] for event in events]

    assert event_names[-1] == "error"
    assert "token" not in event_names
    assert "grounding" not in event_names
    assert "done" not in event_names

    if failure == "verification":
        assert unit_of_work.saved is None
