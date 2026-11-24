from agentiq.config import Settings
from agentiq.research.models import ResearchEvidence, SourceCandidate
from agentiq.research.service import ResearchService


def make_service() -> ResearchService:
    return ResearchService(
        search_provider=None,  # type: ignore[arg-type]
        settings=Settings(
            _env_file=None,
            exa_api_key="test",
        ),
    )


def make_evidence() -> ResearchEvidence:
    return ResearchEvidence(
        query="How should I learn Rust?",
        sources=[
            SourceCandidate(
                id="source:1",
                title="The Rust Book",
                url="https://doc.rust-lang.org/book/",
                excerpt="Rust is a systems programming language.",
            ),
        ],
    )


def test_valid_citation_is_grounded() -> None:
    service = make_service()

    validation = service.validate_citations(
        content="Rust is a systems programming language. [1]",
        evidence=make_evidence(),
    )

    assert validation.is_grounded
    assert validation.cited_source_ids == ["source:1"]
    assert validation.invalid_source_ids == []


def test_invalid_citation_is_removed() -> None:
    service = make_service()

    content = "Claim. [1] [9]"

    validation = service.validate_citations(
        content=content,
        evidence=make_evidence(),
    )

    sanitized = service.sanitize_content(
        content=content,
        validation=validation,
    )

    assert not validation.is_grounded
    print("sanitized: ", sanitized)
    print("validation: ", validation)
    assert validation.invalid_source_ids == ["source:9"]
    assert "[9]" not in sanitized
    assert "[1]" in sanitized


def test_missing_citation_is_not_grounded() -> None:
    service = make_service()

    validation = service.validate_citations(
        content="A factual claim without a citation.",
        evidence=make_evidence(),
    )

    assert not validation.is_grounded
    assert validation.cited_source_ids == []


def test_prompt_preserves_non_contiguous_source_ids() -> None:
    service = make_service()

    evidence = ResearchEvidence(
        query="What does the evidence say?",
        sources=[
            SourceCandidate(
                id="source:1",
                title="First source",
                url="https://example.com/first",
                excerpt="The first supporting excerpt.",
            ),
            SourceCandidate(
                id="source:3",
                title="Third source",
                url="https://example.com/third",
                excerpt="The third supporting excerpt.",
            ),
        ],
    )

    messages = service.build_messages(
        query=evidence.query,
        evidence=evidence,
    )

    prompt = messages[-1]["content"]

    assert "[source:1]\nTitle: First source" in prompt
    assert "[source:3]\nTitle: Third source" in prompt
    assert "[source:2]" not in prompt

    validation = service.validate_citations(
        content="The third supporting excerpt. [3]",
        evidence=evidence,
    )

    assert validation.cited_source_ids == ["source:3"]
    assert validation.invalid_source_ids == []
