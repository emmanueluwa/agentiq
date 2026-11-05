from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentiq.api.routes import router
from agentiq.config import get_settings
from agentiq.logging import configure_logging

configure_logging()

settings = get_settings()

app = FastAPI(
    title="AgentIQ",
    version="0.1.0",
    description="An engineering mentor and product-building advisor.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
