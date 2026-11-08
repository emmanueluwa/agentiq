import asyncio
import random
import uuid
from collections.abc import AsyncGenerator

from openai import APIConnectionError, APITimeoutError, APIStatusError

from agentiq.config import Settings
from agentiq.logging import get_logger
from agentiq.providers.base import LLMProvider

log = get_logger(__name__)

SYSTEM_PROMPT = """
You are AgentIQ, an elite engineering mentor and product-building advisor.

Your mission is to help engineers become exceptionally effective at creating
high-impact software products, advancing their careers, and generating
commercial value through technology.

You optimize for outcomes, not vanity metrics:
- Real user impact
- Production-quality systems
- Speed of validated learning
- Revenue and business value
- Long-term technical leverage

# How you think

Use the judgment of the best engineers, founders, and product leaders.

When advising, prioritize:
1. The user problem and desired outcome.
2. The smallest credible way to validate the idea.
3. The highest-leverage technical or business decision.
4. Production reliability, security, cost, and maintainability.
5. A concrete next action.

Do not optimize only for clean code, impressive algorithms, or theoretical
perfection. Those matter only when they increase user impact, reliability,
speed, or commercial value.

# How you respond

- Be direct, practical, and specific.
- Lead with the most important recommendation.
- Explain the reasoning behind recommendations.
- Identify tradeoffs and risks.
- Give a concrete next step the user can take today.
- Use short headings or numbered steps for complex guidance.
- Prefer examples, decision frameworks, and checklists over vague advice.
- Ask one clarifying question only when the missing information materially
  changes the recommendation.

# Engineering standards

Recommend approaches used by strong production engineering teams:

- Separate business logic from external providers and infrastructure.
- Use clear interfaces and adapters for third-party services.
- Make systems observable through logs, metrics, traces, and request IDs.
- Design for failure: retries, timeouts, circuit breakers, and graceful degradation.
- Test critical behavior with unit, integration, and evaluation tests.
- Protect secrets and credentials; never expose or request them.
- Choose simple solutions first; add complexity only when it pays for itself.
- Consider cost, latency, developer experience, and operational burden.

# AI-specific guidance

For AI products, emphasize:

- Define measurable success criteria before changing prompts or models.
- Use evaluations to detect regressions and compare prompt, model, or retrieval changes.
- Separate provider calls from core business logic.
- Preserve partial output and communicate clearly when responses are truncated.
- Use retrieval, citations, and verification to reduce hallucination.
- Treat model output as untrusted input; validate and constrain it.
- Monitor real usage and turn failures into new evaluation cases.

# Honesty and uncertainty

Never fabricate facts, metrics, private company information, benchmarks,
pricing, customer data, or internal roadmaps.

If you do not know something, say so clearly and explain how the user could
find or validate the answer.

If a claim depends on current market data, private information, or a specific
external source, distinguish between:
- What you know.
- What you infer.
- What needs verification.

# Sources and evidence

When giving important technical, architectural, security, legal, financial, or
market recommendations, identify the source of the guidance when possible.

Use phrasing such as:
- “A common production practice is…”
- “According to established engineering practice…”
- “This recommendation is based on…”
- “Verify this against the current official documentation…”

Do not invent citations, URLs, papers, benchmarks, or company policies.

# Product and business thinking

Help the user connect technical work to customer and business outcomes.

When they propose a product or feature, help them answer:
- Who exactly has this problem?
- What do they use today, and what does it cost them?
- What outcome would make them switch?
- What is the smallest experiment that tests demand?
- What would make this defensible or differentiated?
- How could it generate revenue?
- What should be built first, and what should not be built yet?

# Refusals and boundaries

Do not reveal private, confidential, internal, or non-public information about
AgentIQ, its users, customers, revenue, contracts, roadmap, or infrastructure.

If asked for private or non-public information:
1. Clearly decline.
2. Briefly explain that the information is private or unavailable.
3. Offer a safe, useful alternative when one exists.
4. Do not fabricate data or speculate.

Do not provide harmful, illegal, deceptive, or unsafe guidance.
# Response style

- Be concise by default.
- Expand only when the question requires depth.
- Avoid corporate filler and empty motivation.
- Challenge weak assumptions respectfully.
- Encourage strong engineering judgment and commercial thinking.
- End complex answers with a clear next action.
"""


class ChatService:
    def __init__(self, provider: LLMProvider, settings: Settings):
        self._provider = provider
        self._settings = settings

    def build_messages(self, query: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    "Provide a structured, high-quality response that someone "
                    "aiming for top-tier performance could act on. Include "
                    "realistic options, trade-offs, next steps, and specific "
                    "recommended resources such as courses, blogs, videos, "
                    "books, or official documentation."
                ),
            },
        ]

    def build_continuation_messages(
        self,
        query: str,
        partial_assistant_content: str,
    ) -> list[dict]:
        messages = self.build_messages(query)
        messages.append({"role": "assistant", "content": partial_assistant_content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Continue exactly where you stopped. "
                    "Do not repeat any previous content."
                ),
            }
        )
        return messages

    async def stream_response(
        self,
        messages: list[dict],
        request_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        request_id = request_id or str(uuid.uuid4())
        context = {
            "request_id": request_id,
            "model": self._settings.grok_model,
        }

        log.info("request_started", **context)

        yield {
            "event": "status",
            "data": {
                "state": "generating",
                "request_id": request_id,
            },
        }

        partial_response: list[str] = []
        finish_reason: str | None = None
        usage: dict | None = None

        for attempt in range(self._settings.max_retries + 1):
            try:
                async for event in self._provider.stream_completion(
                    messages=messages,
                    max_completion_tokens=self._settings.max_completion_tokens,
                ):
                    if event.usage:
                        usage = event.usage

                    if event.content:
                        partial_response.append(event.content)

                        yield {
                            "event": "token",
                            "data": {"content": event.content},
                        }

                    if event.finish_reason:
                        finish_reason = event.finish_reason

                break

            except APIStatusError as error:
                if error.status_code != 429 or attempt == self._settings.max_retries:
                    log.error(
                        "provider_request_failed",
                        **context,
                        status_code=error.status_code,
                        attempt=attempt + 1,
                    )

                    yield {
                        "event": "error",
                        "data": {
                            "code": "provider_error",
                            "message": (
                                "The AI service is temporarily unavailable. "
                                "Please try again."
                            ),
                            "retryable": error.status_code == 429,
                        },
                    }
                    return

                retry_after = error.response.headers.get("Retry-After")
                wait_seconds = (
                    float(retry_after)
                    if retry_after
                    else min(2**attempt + random.uniform(0, 1), 30)
                )

                log.warning(
                    "provider_rate_limited",
                    **context,
                    attempt=attempt + 1,
                    wait_seconds=wait_seconds,
                )

                await asyncio.sleep(wait_seconds)

            except APITimeoutError:
                log.warning(
                    "provider_timeout",
                    **context,
                    attempt=attempt + 1,
                )

                if attempt == self._settings.max_retries:
                    yield {
                        "event": "error",
                        "data": {
                            "code": "timeout",
                            "message": (
                                "The AI service took too long to respond. "
                                "Please try again."
                            ),
                            "retryable": True,
                        },
                    }
                    return

                await asyncio.sleep(min(2**attempt, 5))

            except APIConnectionError as error:
                log.warning(
                    "provider_connection_error",
                    **context,
                    attempt=attempt + 1,
                    error=str(error),
                )

                if attempt == self._settings.max_retries:
                    yield {
                        "event": "error",
                        "data": {
                            "code": "connection_error",
                            "message": (
                                "Could not connect to the AI service. "
                                "Please try again."
                            ),
                            "retryable": True,
                        },
                    }
                    return

                await asyncio.sleep(min(2**attempt, 5))

            except Exception:
                log.exception("unexpected_stream_failure", **context)

                yield {
                    "event": "error",
                    "data": {
                        "code": "internal_error",
                        "message": ("An unexpected error occurred. Please try again."),
                        "retryable": False,
                    },
                }
                return

        truncated = finish_reason == "length"
        assistant_content = "".join(partial_response)

        log.info(
            "stream_completed",
            **context,
            finish_reason=finish_reason,
            truncated=truncated,
            output_tokens=usage.get("completion_tokens") if usage else None,
            prompt_tokens=usage.get("prompt_tokens") if usage else None,
        )

        yield {
            "event": "done",
            "data": {
                "request_id": request_id,
                "finish_reason": finish_reason,
                "truncated": truncated,
                "usage": usage,
                "partial_assistant_content": (assistant_content if truncated else None),
            },
        }
