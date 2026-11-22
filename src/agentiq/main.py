from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentiq.api.routes import router
from agentiq.api.routes_learning import router as learning_router
from agentiq.config import get_settings
from agentiq.learning.database import (
    create_database_engine,
    create_session_factory,
)
from agentiq.logging import configure_logging

configure_logging()

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)

app.include_router(
    learning_router,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
