import asyncio
import json

import pytest

from agentiq.providers.base import LLMProvider, StreamEvent
from agentiq.research.models import ResearchEvidence, SourceCandidate
from agentiq.research.verification import (
    EvidenceVerifier,
    VerificationError,
)

EXCERPT = "Rust is a systems programming language."
CONTENT = f"{EXCERPT} [1]"


class FakeProvider(LLMProvider):
    def __init__(self, report, *, finish_reason="stop"):
        self.report = report
        self.finish_reason = finish_reason

    async def stream_completion(self, messages, max_completion_tokens):
        yield StreamEvent(content=json.dumps(self.report))
        yield StreamEvent(finish_reason=self.finish_reason)


def make_evidence():
    return ResearchEvidence(
        query="What kind of language is Rust?",
        sources=[
            SourceCandidate(
                id="source:1",
                title="Example source",
                url="https://example.com/rust",
                excerpt=EXCERPT,
            )
        ],
    )


def make_report(
    *,
    verdict="supported",
    evidence_sufficient=True,
    line_id=0,
    source_id="source:1",
    quote=EXCERPT,
):
    return {
        "evidence_sufficient": evidence_sufficient,
        "reviews": [
            {
                "line_id": line_id,
                "verdict": verdict,
                "explanation": "Test review.",
                "supporting_quotes": (
                    [{"source_id": source_id, "quote": quote}]
                    if verdict == "supported"
                    else []
                ),
            }
        ],
    }


def verify(report, *, content=CONTENT, finish_reason="stop"):
    verifier = EvidenceVerifier(FakeProvider(report, finish_reason=finish_reason))

    return asyncio.run(
        verifier.verify(
            content=content,
            evidence=make_evidence(),
        )
    )


def test_accepts_internally_consistent_supported_report():
    result = verify(make_report())

    assert result.accepted


@pytest.mark.parametrize("verdict", ["unsupported", "non_factual"])
def test_does_not_accept_unsupported_or_only_non_factual_answers(verdict):
    result = verify(make_report(verdict=verdict))

    assert not result.accepted


def test_rejects_answer_when_evidence_is_insufficient():
    result = verify(make_report(evidence_sufficient=False))

    assert not result.accepted


@pytest.mark.parametrize(
    "report",
    [
        make_report(line_id=1),
        make_report(source_id="source:9"),
        make_report(quote="A quotation that was never retrieved."),
        {"evidence_sufficient": True, "reviews": []},
        {"evidence_sufficient": "true", "reviews": []},
        {"evidence_sufficient": True, "reviews": [], "unexpected": True},
    ],
)
def test_rejects_invalid_reports(report):
    with pytest.raises(VerificationError):
        verify(report)


def test_rejects_report_that_omits_a_draft_line():
    with pytest.raises(VerificationError):
        verify(
            make_report(),
            content=f"{CONTENT}\nAnother factual statement.",
        )


def test_rejects_duplicate_line_reviews():
    report = make_report()
    report["reviews"].append(dict(report["reviews"][0]))

    with pytest.raises(VerificationError):
        verify(report)


def test_rejects_supported_line_without_a_citation():
    with pytest.raises(VerificationError):
        verify(make_report(), content=EXCERPT)


def test_rejects_supported_line_with_an_unknown_citation():
    with pytest.raises(VerificationError):
        verify(make_report(), content=f"{CONTENT} [9]")


def test_rejects_truncated_verifier_output():
    with pytest.raises(VerificationError):
        verify(make_report(), finish_reason="length")


def test_rejects_malformed_json():
    class InvalidJsonProvider(LLMProvider):
        async def stream_completion(self, messages, max_completion_tokens):
            yield StreamEvent(content="This is not JSON.")
            yield StreamEvent(finish_reason="stop")

    verifier = EvidenceVerifier(InvalidJsonProvider())

    with pytest.raises(VerificationError):
        asyncio.run(
            verifier.verify(
                content=CONTENT,
                evidence=make_evidence(),
            )
        )


def test_verification_has_a_deadline():
    class SlowProvider(LLMProvider):
        async def stream_completion(self, messages, max_completion_tokens):
            await asyncio.sleep(1)
            yield StreamEvent(content="unreachable")

    verifier = EvidenceVerifier(
        SlowProvider(),
        timeout_seconds=0.01,
    )

    with pytest.raises(VerificationError, match="timed out"):
        asyncio.run(
            verifier.verify(
                content=CONTENT,
                evidence=make_evidence(),
            )
        )


def test_cancellation_is_not_converted_into_verification_failure():
    class CancelledProvider(LLMProvider):
        async def stream_completion(self, messages, max_completion_tokens):
            raise asyncio.CancelledError()
            yield StreamEvent()

    verifier = EvidenceVerifier(CancelledProvider())

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            verifier.verify(
                content=CONTENT,
                evidence=make_evidence(),
            )
        )
