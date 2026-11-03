import os
import structlog
import logging
import random
import asyncio
import uuid
from collections.abc import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, APIStatusError
from dotenv import load_dotenv
import httpx

load_dotenv()


structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "grok-4-1-fast"
MAX_COMPLETION_TOKENS = 4096
MAX_RETRIES = 3

client = AsyncOpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(read=180.0, connect=10.0, write=30.0, pool=30.0),
    max_retries=1,  # streaming so avoid duplpicating partial output
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)


SYSTEM_PROMPT = """You are an exceptionally knowledgeable mentor with deep expertise across domains. You operate at the level of the top 1% of practitioners in whatever field the user is asking about.

Your job is to give clear, rigorous, and practical guidance.

Rules you must follow:
- Be precise and evidence-based. State assumptions explicitly.
- Structure your response with clear headings.
- Always include: key trade-offs or risks, realistic timelines or effort estimates where relevant, and a short summary of recommended next steps.
- Prefer concrete examples, numbers, ranges, and specific techniques over vague advice.
- If the query is vague, make reasonable assumptions and state them.
- Adapt your depth and terminology to the user’s apparent level, but never dumb things down unnecessarily.
- Prefer clarity and usefulness over length.
- Always recommend high-quality resources (courses, books, blogs, YouTube channels/videos, official docs, or communities) that are relevant to the topic. Prefer well-known, respected sources.
"""


def build_messages(user_query: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Question:
{user_query}

Provide a structured, high-quality response that someone aiming for top-tier performance could act on. Include realistic options, trade-offs, next steps, and specific recommended resources (courses, blogs, YouTube videos, books, or official documentation).""",
        },
    ]


# server sent events for real time updates
def sse_event(event: str, data: dict) -> str:
    import json

    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@app.post("/")
async def root(request: Request, body: QueryRequest):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request_context = {"request_id": request_id, "model": MODEL}

    log.info("request_started", **request_context)

    if not os.getenv("GROK_API_KEY"):
        log.error("missing_api_key", **request_context)
        raise HTTPException(
            status_code=500, detail="XAI_API_KEY environment variable is missing."
        )

    formatted_messages = build_messages(body.query)

    return StreamingResponse(
        stream_grok_response(formatted_messages, request_context=request_context),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


async def stream_grok_response(
    messages: list[dict], request_context: dict
) -> AsyncGenerator[str, None]:
    """
    yielding text chunks from Grok as they arrive
    """
    request_id = request_context["request_id"]

    yield sse_event(
        "status",
        {
            "state": "generating",
            "request_id": request_id,
        },
    )

    partial_response: list[str] = []
    finish_reason: str | None = None
    usage: dict | None = None

    log.info("stream_started", model=MODEL, message_count=len(messages))

    for attempt in range(MAX_RETRIES + 1):

        try:
            # requesting streaming response from xAI endpoint
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if not chunk.choices:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage.model_dump()
                    continue

                choice = chunk.choices[0]

                if choice.delta and choice.delta.content:
                    content = choice.delta.content
                    partial_response.append(content)

                    yield sse_event("token", {"content": content})

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            break

        except APIStatusError as error:
            if error.status_code != 429 or attempt == MAX_RETRIES:
                log.error(
                    "provider_request_failed",
                    **request_context,
                    status_code=error.status_code,
                    attempt=attempt + 1,
                )

                yield sse_event(
                    "error",
                    {
                        "code": "provider_error",
                        "message": "The AI service is temporarily unavailable. Please try again.",
                        "retryable": error.status_code == 429,
                    },
                )
                return

            retry_after = error.response.headers.get("Retry-After")
            wait_seconds = (
                float(retry_after)
                if retry_after
                else min(2**attempt + random.uniform(0, 1), 30)
            )

            log.warning(
                "provider_rate_limited",
                **request_context,
                attempt=attempt + 1,
                wait_seconds=wait_seconds,
            )

            await asyncio.sleep(wait_seconds)

        except APITimeoutError:
            log.warning(
                "provider_timeout",
                **request_context,
                attempt=attempt + 1,
            )

            if attempt == MAX_RETRIES:
                yield sse_event(
                    "error",
                    {
                        "code": "timeout",
                        "message": "The AI service took too long to respond. Please try again.",
                        "retryable": True,
                    },
                )
                return

            await asyncio.sleep(min(2**attempt, 5))

        except APIConnectionError as error:
            log.warning(
                "provider_connection_error",
                **request_context,
                attempt=attempt + 1,
                error=str(error),
            )

            if attempt == MAX_RETRIES:
                yield sse_event(
                    "error",
                    {
                        "code": "connection_error",
                        "message": "Could not connect to the AI service. Please try again.",
                        "retryable": True,
                    },
                )
                return

            await asyncio.sleep(min(2**attempt, 5))

        except Exception as error:
            log.exception(
                "unexpected_stream_failure",
                **request_context,
            )

            yield sse_event(
                "error",
                {
                    "code": "internal_error",
                    "message": "An unexpected error occurred. Please try again.",
                    "retryable": False,
                },
            )
            return

    truncated = finish_reason == "length"
    assistant_content = "".join(partial_response)

    log.info(
        "stream_completed",
        **request_context,
        finish_reason=finish_reason,
        truncated=truncated,
        output_tokens=usage.get("completion_tokens") if usage else None,
        prompt_tokens=usage.get("prompt_tokens") if usage else None,
    )

    yield sse_event(
        "done",
        {
            "request_id": request_id,
            "finish_reason": finish_reason,
            "truncated": truncated,
            "usage": usage,
            "partial_assistant_content": assistant_content if truncated else None,
        },
    )


class ContinueRequest(BaseModel):
    query: str
    partial_assistant_content: str = Field(..., min_length=1)


@app.post("/continue")
async def continue_chat(
    request: Request,
    body: ContinueRequest,
) -> StreamingResponse:
    messages = build_messages(body.query)

    messages.append(
        {
            "role": "assistant",
            "content": body.partial_assistant_content,
        }
    )

    messages.append(
        {
            "role": "user",
            "content": (
                "Continue exactly where you stopped. "
                "Do not repeat any previous content."
            ),
        }
    )

    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

    return StreamingResponse(
        stream_grok_response(
            messages=messages,
            request_context={
                "request_id": request_id,
                "model": MODEL,
                "continuation": True,
            },
        ),
        media_type="text/event-stream",
    )
