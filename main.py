import os
import structlog
import logging
from fastapi import FastAPI, HTTPException, APITimeoutError, APIExceptionError
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
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

TASK = "I want to learn about about robotics"

input_prompt = f"""
        {TASK}. Respond with a step by step plan, link to youtube video to watch then questions/tasks to answer/complete. 
        The questions should pogressively increase in difficulty. 2 questions
        """


@app.get("/models")
async def list_models():
    models = await client.models.list()
    return [m.id for m in models.data]


@app.get("/")
async def root():
    log.info("request_started", path="/", model="grok-4.7")

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
    log.info("stream_started", model="grok-4.7", message_count=len(messages))

    try:
        # requesting streaming response from xAI endpoint
        stream = await client.chat.completions.create(
            model="grok-4.7", messages=messages, stream=True
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

        log.info("stream_completed", model="grok-4.7")

    except APITimeoutError:
        log.warning("stream_timeout", model="grok-4.7")
        yield "\n[Request timed out. Please try again.]"

    except APIExceptionError as e:
        log.warning("stream_connection_error", error=str(e), model="grok-4.7")
        yield "\n[Connection error. Please try again.]"

    except Exception as e:
        log.error("stream_failed", error=str(e), model="grok-4.7")
        yield f"\n[An error occured. Please try again.]"
