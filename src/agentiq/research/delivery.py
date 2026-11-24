from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from uuid import UUID

from agentiq.chat.unit_of_work import SqlChatUnitOfWork
from agentiq.logging import get_logger
from agentiq.research.draft import DraftGenerationError, collect_draft
from agentiq.research.models import ResearchEvidence
from agentiq.research.service import ResearchService
from agentiq.research.verification import VerificationError
from agentiq.services.chat_service import ChatService

log = get_logger(__name__)

UNVERIFIED_ANSWER_RESPONSE = (
    "I don’t have enough verified evidence to answer this confidently. "
    "I could not substantiate the generated answer against the retrieved "
    "source excerpts, so I have not presented it. "
    "Please narrow the question or provide additional reliable sources."
)


async def deliver_grounded_answer(
    *,
    messages: list[dict],
    evidence: ResearchEvidence,
    chat_service: ChatService,
    research_service: ResearchService,
    unit_of_work_factory: Callable[[], SqlChatUnitOfWork],
    request_id: str,
    conversation_id: UUID,
    assistant_message_id: UUID,
) -> AsyncGenerator[dict, None]:
    identifiers = {
        "request_id": request_id,
        "conversation_id": str(conversation_id),
        "assistant_message_id": str(assistant_message_id),
    }

    yield {
        "event": "status",
        "data": {
            **identifiers,
            "state": "generating",
        },
    }

    try:
        draft = await collect_draft(
            chat_service.stream_response(
                messages=messages,
                **identifiers,
            )
        )
    except DraftGenerationError:
        log.exception("draft_generation_failed", **identifiers)

        yield {
            "event": "error",
            "data": {
                "code": "draft_generation_failed",
                "message": (
                    "The answer could not be generated successfully. "
                    "No unverified answer has been released. Please try again."
                ),
                "retryable": True,
            },
        }
        return
    except Exception:
        log.exception("unexpected_draft_failure", **identifiers)

        yield {
            "event": "error",
            "data": {
                "code": "draft_generation_failed",
                "message": (
                    "Answer generation failed. "
                    "No unverified answer has been released. Please try again."
                ),
                "retryable": True,
            },
        }
        return

    yield {
        "event": "status",
        "data": {
            **identifiers,
            "state": "verifying",
        },
    }

    citations = research_service.validate_citations(
        content=draft.content,
        evidence=evidence,
    )

    accepted = False

    if citations.has_citations and not citations.invalid_source_ids:
        try:
            report = await chat_service.verify_evidence(
                content=draft.content,
                evidence=evidence,
            )
            accepted = report.accepted
        except VerificationError:
            log.exception("answer_verification_failed", **identifiers)

            yield {
                "event": "error",
                "data": {
                    "code": "verification_failed",
                    "message": (
                        "I could not complete the evidence check. "
                        "No unverified answer has been released. Please try again."
                    ),
                    "retryable": True,
                },
            }
            return

    content = draft.content if accepted else UNVERIFIED_ANSWER_RESPONSE
    truncated = draft.truncated if accepted else False
    cited_source_ids = citations.cited_source_ids if accepted else []

    # These describe the released answer, not the rejected private draft.
    sources = evidence.sources if accepted else []

    source_payload = [
        {
            "source_number": int(source.id.split(":")[1]),
            "source_id": source.id,
            "title": source.title,
            "url": str(source.url),
            "source_type": source.source_type,
            "published_at": source.published_at,
            "excerpt": source.excerpt,
            "relevance": source.relevance,
        }
        for source in sources
    ]

    try:
        async with unit_of_work_factory() as unit_of_work:
            await unit_of_work.update_message(
                message_id=assistant_message_id,
                content=content,
                truncated=truncated,
                cited_source_ids=cited_source_ids,
                invalid_source_ids=[],
                grounded=accepted,
                sources=source_payload,
            )
    except Exception:
        log.exception("verified_answer_persistence_failed", **identifiers)

        yield {
            "event": "error",
            "data": {
                "code": "persistence_failed",
                "message": (
                    "The response could not be saved, so it has not been "
                    "released. Please try again."
                ),
                "retryable": True,
            },
        }
        return

    log.info(
        "answer_release_decided",
        **identifiers,
        accepted=accepted,
        truncated=truncated,
        cited_source_count=len(cited_source_ids),
        invalid_draft_citation_count=len(citations.invalid_source_ids),
    )

    yield {
        "event": "grounding",
        "data": {
            "cited_source_ids": cited_source_ids,
            "invalid_source_ids": [],
            "grounded": accepted,
            "sources": [
                {
                    "id": source.id,
                    "title": source.title,
                    "url": str(source.url),
                    "source_type": source.source_type,
                    "published_at": (
                        source.published_at.isoformat() if source.published_at else None
                    ),
                    "excerpt": source.excerpt,
                }
                for source in sources
            ],
        },
    }

    yield {
        "event": "token",
        "data": {"content": content},
    }

    yield {
        "event": "done",
        "data": {
            **identifiers,
            "finish_reason": draft.finish_reason if accepted else "stop",
            "truncated": truncated,
            "usage": draft.usage,
            "partial_assistant_content": content if truncated else None,
        },
    }
