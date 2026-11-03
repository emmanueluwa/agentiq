import os
import structlog
import logging
import random
import asyncio
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

client = AsyncOpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(180.0, connect=10.0, write=30.0, pool=30.0),
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


@app.post("/")
async def root(request: Request, body: QueryRequest):
    log.info("request_started", path="/", model="grok-4-1-fast")

    if not os.getenv("GROK_API_KEY"):
        log.error("missing_api_key")
        raise HTTPException(
            status_code=500, detail="XAI_API_KEY environment variable is missing."
        )

    formatted_messages = build_messages(body.query)

    return StreamingResponse(
        stream_grok_response(formatted_messages), media_type="text/plain"
    )


async def stream_grok_response(messages: list[dict]):
    """
    yielding text chunks from Grok as they arrive
    """
    log.info("stream_started", model="grok-4-1-fast", message_count=len(messages))

    max_retries = 3

    for attempt in range(max_retries + 1):

        try:
            yield "Generating finance research report... \n\n"

            # requesting streaming response from xAI endpoint
            stream = await client.chat.completions.create(
                model="grok-4-1-fast", messages=messages, stream=True
            )

            first_chunk = True
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    if first_chunk:
                        log.info("first_chunk_recieved", model="grok-4-1-fast")
                        first_chunk = False
                    yield content

            log.info("stream_completed", model="grok-4-1-fast")

            return  # exiting retry loop

        except APIStatusError as e:
            if e.status_code == 429:
                if attempt == max_retries:
                    log.warning("rate_limit_exhausted", attempts=attempt + 1)

                    yield "\n\n[Too many requests. Please try again in a minute.]"
                    return

                # adhere to retry after if server sent it
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    wait = float(retry_after)
                else:
                    # fallback: exceptional backoff + jitter
                    wait = min(2**attempt + random.uniform(0, 1), 30)

                log.warning(
                    "rate_limited",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                    model="grok-4-1-fast",
                )
                await asyncio.sleep(wait)
                continue

            ## no retry for any other status error
            log.error("stream_api_error", status=e.status_code, error=str(e))
            yield "\n\n[An error occured. Please try again.]"
            return

        except APITimeoutError:
            log.warning("stream_timeout", model="grok-4-1-fast")
            yield "\n\n[Request timed out. Please try again.]"

        except APIConnectionError as e:
            log.warning("stream_connection_error", error=str(e), model="grok-4-1-fast")
            yield "\n\n[Connection error. Please try again.]"

        except Exception as e:
            log.error("stream_failed", error=str(e), model="grok-4-1-fast")
            yield f"\n\n[An error occured. Please try again.]"
