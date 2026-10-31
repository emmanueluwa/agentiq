import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = AsyncOpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1",
)

TASK = "I want to learn about linked lists, for the purpose of being able to solve the problem 'Merge Two Sorted Lists' on leedcode"

input = f"""
        {TASK}. Respond with a step by step plan, link to youtube video to watch then questions/tasks to answer/complete. 
        The questions should pogressively icrease in difficulty. 4 questions
        """


@app.get("/models")
async def list_models():
    models = await client.models.list()
    return [m.id for m in models.data]


@app.get("/")
async def root():
    if not os.getenv("GROK_API_KEY"):
        raise HTTPException(
            status_code=500, detail="XAI_API_KEY environment variable is missing."
        )

    formatted_messages = [{"role": "system", "content": input}]

    return StreamingResponse(
        stream_grok_response(formatted_messages), media_type="text/plain"
    )


async def stream_grok_response(messages: list[dict]):
    """
    yielding text chunks from grok as they arrive
    """
    try:
        # requesting streaming response from xAI endpoint
        stream = await client.chat.completions.create(
            model="grok-4.7", messages=messages, stream=True
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        yield f"\n[API Error: {str(e)}]"
