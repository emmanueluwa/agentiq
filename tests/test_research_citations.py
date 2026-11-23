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
