import asyncio

import pytest

from agentiq.research.draft import (
    DraftGenerationError,
    collect_draft,
)


async def emit(events):
    for event in events:
        yield event


def token(content):
    return {
        "event": "token",
        "data": {"content": content},
    }


def done(*, truncated=False, finish_reason="stop"):
    return {
        "event": "done",
        "data": {
            "truncated": truncated,
            "finish_reason": finish_reason,
            "usage": None,
        },
    }


def test_collects_draft_without_emitting_client_events():
    draft = asyncio.run(
        collect_draft(
            emit(
                [
                    {
                        "event": "status",
                        "data": {"state": "generating"},
                    },
                    token("Supported "),
                    token("statement. [1]"),
                    done(),
                ]
            )
        )
    )

    assert draft.content == "Supported statement. [1]"
    assert draft.truncated is False
    assert draft.finish_reason == "stop"
    assert draft.usage is None


@pytest.mark.parametrize(
    "events",
    [
        [],
        [token("An unfinished response.")],
        [done()],
        [token("   "), done()],
        [
            token("Partial text."),
            {
                "event": "error",
                "data": {"message": "Provider failed."},
            },
        ],
        [token("Text."), done(), token("Unexpected extra text.")],
        [token("Text."), done(), done()],
        [
            token("Text."),
            done(truncated=False, finish_reason="length"),
        ],
        [
            token("Text."),
            done(truncated=True, finish_reason="stop"),
        ],
        [
            token("Text."),
            done(finish_reason="content_filter"),
        ],
    ],
)
def test_rejects_unusable_or_inconsistent_generation(events):
    with pytest.raises(DraftGenerationError):
        asyncio.run(collect_draft(emit(events)))


def test_preserves_truncation_for_later_policy_decision():
    draft = asyncio.run(
        collect_draft(
            emit(
                [
                    token("An incomplete draft"),
                    done(truncated=True, finish_reason="length"),
                ]
            )
        )
    )

    assert draft.content == "An incomplete draft"
    assert draft.truncated is True
    assert draft.finish_reason == "length"


def test_propagates_unexpected_stream_exceptions():
    async def broken_stream():
        yield token("Partial text.")
        raise RuntimeError("Connection interrupted")

    with pytest.raises(RuntimeError, match="Connection interrupted"):
        asyncio.run(collect_draft(broken_stream()))
