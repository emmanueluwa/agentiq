from collections.abc import AsyncGenerator
from dataclasses import dataclass


@dataclass
class StreamEvent:
    content: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None


class LLMProvider:
    async def stream_completion(
        self,
        messages: list[dict],
        max_completion_tokens: int,
    ) -> AsyncGenerator[StreamEvent, None]:
        raise NotImplementedError
