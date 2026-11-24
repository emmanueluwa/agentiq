from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any


class DraftGenerationError(Exception):
    """Raised when generation does not produce a usable draft."""


@dataclass(frozen=True)
class GeneratedDraft:
    content: str
    truncated: bool
    finish_reason: str
    usage: dict[str, Any] | None


async def collect_draft(
    events: AsyncIterable[dict[str, Any]],
) -> GeneratedDraft:
    """Collect generation events without releasing draft text to a client."""
    content_parts: list[str] = []
    completion: dict[str, Any] | None = None

    async for event in events:
        event_type = event.get("event")

        if completion is not None:
            raise DraftGenerationError("Generation emitted events after completion.")

        if event_type == "token":
            content = event.get("data", {}).get("content")

            if not isinstance(content, str):
                raise DraftGenerationError("Generation returned an invalid text chunk.")

            content_parts.append(content)

        elif event_type == "error":
            raise DraftGenerationError(
                "Generation failed before a draft could be accepted."
            )

        elif event_type == "done":
            completion = event.get("data")

            if not isinstance(completion, dict):
                raise DraftGenerationError(
                    "Generation returned invalid completion metadata."
                )

    if completion is None:
        raise DraftGenerationError("Generation ended without a completion event.")

    content = "".join(content_parts)

    if not content.strip():
        raise DraftGenerationError("Generation returned an empty draft.")

    finish_reason = completion.get("finish_reason")

    if finish_reason not in {"stop", "length"}:
        raise DraftGenerationError(
            "Generation ended with an unsupported finish reason."
        )

    truncated = completion.get("truncated")

    if not isinstance(truncated, bool):
        raise DraftGenerationError("Generation returned invalid truncation metadata.")

    if truncated != (finish_reason == "length"):
        raise DraftGenerationError(
            "Generation returned inconsistent completion metadata."
        )

    usage = completion.get("usage")

    if usage is not None and not isinstance(usage, dict):
        raise DraftGenerationError("Generation returned invalid usage metadata.")

    return GeneratedDraft(
        content=content,
        truncated=truncated,
        finish_reason=finish_reason,
        usage=usage,
    )
