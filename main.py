import os
import structlog
import logging
import random
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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

client = AsyncOpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1",
    timeout=httpx.Timeout(180.0, connect=10.0, write=30.0, pool=30.0),
    max_retries=1,  # streaming so avoid duplpicating partial output
)

TASK = "finance investment research"

input_prompt = """
You are a practical finance research assistant.
Rank the top 4 investment options right now and for each give one article link and one YouTube video link.
"""


@app.get("/models")
async def list_models():
    models = await client.models.list()
    return [m.id for m in models.data]


@app.get("/")
async def root():
    log.info("request_started", path="/", model="grok-4-1-fast")

    if not os.getenv("GROK_API_KEY"):
        log.error("missing_api_key")
        raise HTTPException(
            status_code=500, detail="XAI_API_KEY environment variable is missing."
        )

    formatted_messages = [{"role": "system", "content": input_prompt}]

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
                    log.warning("rate_limit_exhausred", attempts=attempt + 1)

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
