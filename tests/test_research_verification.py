import asyncio
import json

import pytest

from agentiq.providers.base import LLMProvider, StreamEvent
from agentiq.research.models import ResearchEvidence, SourceCandidate
from agentiq.research.verification import (
    EvidenceVerifier,
    VerificationError,
    apply_verified_citations,
)

EXCERPT = "Rust is a systems programming language."
CONTENT = f"{EXCERPT} [1]"


class FakeProvider(LLMProvider):
    def __init__(self, report, *, finish_reason="stop"):
        self.report = report
        self.finish_reason = finish_reason
        self.payloads = []

    async def stream_completion(self, messages, max_completion_tokens):
        self.payloads.append(json.loads(messages[-1]["content"]))

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
    passage_ids=None,
    line_id=0,
):
    if passage_ids is None:
        passage_ids = ["source:1:p1"] if verdict == "supported" else []

    return {
        "evidence_sufficient": evidence_sufficient,
        "reviews": [
            {
                "line_id": line_id,
                "verdict": verdict,
                "explanation": "Test review.",
                "supporting_passage_ids": passage_ids,
            }
        ],
    }


def verify(report, *, content=CONTENT, evidence=None, finish_reason="stop"):
    verifier = EvidenceVerifier(FakeProvider(report, finish_reason=finish_reason))

    return asyncio.run(
        verifier.verify(
            content=content,
            evidence=evidence if evidence is not None else make_evidence(),
        )
    )


def test_selected_passage_resolves_to_original_source_text():
    result = verify(make_report())

    assert result.accepted
    quote = result.reviews[0].supporting_quotes[0]

    assert quote.source_id == "source:1"
    assert quote.quote == EXCERPT


def test_passage_text_is_preserved_without_model_copying():
    evidence = make_evidence()
    evidence.sources[0].excerpt = (
        "First paragraph.\n\n" "A passage with “Unicode”,\n  indentation, and spacing."
    )

    provider = FakeProvider(make_report(passage_ids=["source:1:p2"]))
    verifier = EvidenceVerifier(provider)

    result = asyncio.run(
        verifier.verify(
            content="A claim. [1]",
            evidence=evidence,
        )
    )

    selected = provider.payloads[0]["passages"][1]

    assert selected["id"] == "source:1:p2"
    assert selected["text"] in evidence.sources[0].excerpt
    assert result.reviews[0].supporting_quotes[0].quote == selected["text"]


@pytest.mark.parametrize("verdict", ["unsupported", "non_factual"])
def test_does_not_accept_unsupported_or_only_non_factual_answers(verdict):
    assert not verify(make_report(verdict=verdict)).accepted


def test_insufficient_evidence_prevents_acceptance():
    assert not verify(make_report(evidence_sufficient=False)).accepted


@pytest.mark.parametrize(
    "report",
    [
        make_report(line_id=1),
        make_report(passage_ids=["source:9:p1"]),
        make_report(passage_ids=["source:1:p999"]),
        make_report(passage_ids=[]),
        make_report(passage_ids=["source:1:p1", "source:1:p1"]),
        make_report(
            verdict="unsupported",
            passage_ids=["source:1:p1"],
        ),
        {"evidence_sufficient": True, "reviews": []},
        {"evidence_sufficient": "true", "reviews": []},
        {
            "evidence_sufficient": True,
            "reviews": [],
            "unexpected": True,
        },
    ],
)
def test_rejects_inconsistent_or_invalid_reports(report):
    with pytest.raises(VerificationError):
        verify(report)


def test_rejects_model_written_quotation_field():
    report = make_report()
    report["reviews"][0]["supporting_quotes"] = [
        {
            "source_id": "source:1",
            "quote": "A model-written quotation.",
        }
    ]

    with pytest.raises(VerificationError):
        verify(report)


def test_rejects_missing_line_review():
    with pytest.raises(VerificationError):
        verify(
            make_report(),
            content=f"{CONTENT}\nAnother statement.",
        )


def test_rejects_duplicate_line_reviews():
    report = make_report()
    report["reviews"].append(dict(report["reviews"][0]))

    with pytest.raises(VerificationError):
        verify(report)


def test_empty_evidence_is_not_sent_to_provider():
    provider = FakeProvider(make_report())
    verifier = EvidenceVerifier(provider)

    result = asyncio.run(
        verifier.verify(
            content=CONTENT,
            evidence=ResearchEvidence(query="Question", sources=[]),
        )
    )

    assert not result.accepted
    assert provider.payloads == []


def test_rejects_incomplete_verifier_generation():
    with pytest.raises(VerificationError):
        verify(make_report(), finish_reason="length")


def test_rejects_malformed_json():
    class InvalidJsonProvider(LLMProvider):
        async def stream_completion(self, messages, max_completion_tokens):
            yield StreamEvent(content="Not JSON.")
            yield StreamEvent(finish_reason="stop")

    with pytest.raises(VerificationError):
        asyncio.run(
            EvidenceVerifier(InvalidJsonProvider()).verify(
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


def test_cancellation_propagates():
    class CancelledProvider(LLMProvider):
        async def stream_completion(self, messages, max_completion_tokens):
            raise asyncio.CancelledError()
            yield StreamEvent()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            EvidenceVerifier(CancelledProvider()).verify(
                content=CONTENT,
                evidence=make_evidence(),
            )
        )


def test_supported_line_requires_at_least_one_passage():
    with pytest.raises(VerificationError):
        verify(make_report(passage_ids=[]))


def test_supported_line_citations_are_rewritten_from_selected_passages():
    evidence = make_evidence()
    evidence.sources.append(
        SourceCandidate(
            id="source:2",
            title="Second source",
            url="https://example.com/second",
            excerpt=EXCERPT,
        )
    )

    report = verify(
        make_report(passage_ids=["source:1:p1", "source:2:p1"]),
        content=f"{EXCERPT} [1] [9]",
        evidence=evidence,
    )

    rewritten = apply_verified_citations(
        f"{EXCERPT} [1] [9]",
        report,
    )

    assert rewritten == f"{EXCERPT} [1][2]"


def test_non_supported_line_citations_are_not_rewritten():
    report = verify(
        make_report(verdict="unsupported"),
        content=f"{EXCERPT} [1]",
    )

    rewritten = apply_verified_citations(
        f"{EXCERPT} [1]",
        report,
    )

    assert rewritten == f"{EXCERPT} [1]"


def test_supported_line_with_no_citations_gets_application_citations():
    report = verify(
        make_report(passage_ids=["source:1:p1"]),
        content=EXCERPT,
    )

    rewritten = apply_verified_citations(
        EXCERPT,
        report,
    )

    assert rewritten == f"{EXCERPT} [1]"
