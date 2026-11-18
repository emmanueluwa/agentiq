from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from agentiq.config import settings
from agentiq.learning.database import (
    create_database_engine,
    create_session_factory,
    create_database_engine_with_url,
)
from agentiq.learning.sql_models import Base
from agentiq.learning.sql_repositories import (
    SqlAssessmentEvidenceRepository,
    SqlProfileRepository,
)
from agentiq.learning.sql_unit_of_work import SqlLearningUnitOfWork


@pytest.fixture
async def postgres_engine() -> AsyncIterator[AsyncEngine]:
    if settings.test_database_url is None:
        pytest.skip("TEST_DATABASE_URL is not configured")

    engine = create_database_engine_with_url(settings.test_database_url)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def postgres_session(
    postgres_engine: AsyncEngine,
) -> AsyncIterator[AsyncSession]:
    session_factory = create_session_factory(postgres_engine)

    async with session_factory() as session:
        yield session


@pytest.fixture
def postgres_unit_of_work(
    postgres_session: AsyncSession,
) -> SqlLearningUnitOfWork:
    return SqlLearningUnitOfWork(postgres_session)
