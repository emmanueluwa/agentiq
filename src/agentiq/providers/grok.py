from collections.abc import AsyncGenerator

import httpx
from openai import AsyncOpenAI

from agentiq.config import Settings
from agentiq.providers.base import LLMProvider, StreamEvent


class GrokProvider(LLMProvider):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.grok_api_key,
            base_url=settings.grok_base_url,
            timeout=httpx.Timeout(
                read=180.0,
                connect=10.0,
                write=30.0,
                pool=30.0,
            ),
            max_retries=1,
        )

    async def stream_completion(
        self,
        messages: list[dict],
        max_completion_tokens: int,
    ) -> AsyncGenerator[StreamEvent, None]:
        stream = await self._client.chat.completions.create(
            model=self._settings.grok_model,
            messages=messages,
            stream=True,
            max_completion_tokens=max_completion_tokens,
            stream_options={"include_usage": True},
        )

        async for chunk in stream:
            if not chunk.choices:
                if getattr(chunk, "usage", None):
                    yield StreamEvent(usage=chunk.usage.model_dump())
                continue

            choice = chunk.choices[0]

            yield StreamEvent(
                content=choice.delta.content if choice.delta else None,
                finish_reason=choice.finish_reason,
            )
