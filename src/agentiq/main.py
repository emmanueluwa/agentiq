from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from agentiq.api.routes import router
from agentiq.config import get_settings
from agentiq.logging import configure_logging
from agentiq.api.routes_learning import router as learning_router
from agentiq.learning.database import (
    create_database_engine,
    create_session_factory,
)

configure_logging()

settings = get_settings()


@asynccontextmanager
async def lifespan(app):
    engine = create_database_engine()
    session_factory = create_session_factory(engine)

    app.state.database_engine = engine
    app.state.session_factory = session_factory

    yield

    await engine.dispose()


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
app.include_router(learning_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
