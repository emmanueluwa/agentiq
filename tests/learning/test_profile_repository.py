from __future__ import annotations

import uuid

import pytest

from agentiq.learning.models import (
    LearnerConstraint,
    LearnerProfile,
)
from agentiq.learning.profile_repository import (
    ProfileConcurrencyError,
    InMemoryProfileRepository,
    ProfileNotFoundError,
    ProfileRepository,
    ProfileRepositoryError,
)


def build_profile() -> LearnerProfile:
    return LearnerProfile(
        constraints=LearnerConstraint(
            weekly_hours_available=10,
            preferred_session_minutes=45,
        )
    )


@pytest.mark.asyncio
async def test_save_and_get_profile() -> None:
    repository = InMemoryProfileRepository()
    profile = build_profile()

    saved_profile = await repository.save_profile(profile)
    retrieved_profile = await repository.get_profile(profile.id)

    assert saved_profile.id == profile.id
    assert retrieved_profile.id == profile.id
    assert retrieved_profile.constraints.weekly_hours_available == 10


@pytest.mark.asyncio
async def test_get_missing_profile_raises_not_found() -> None:
    repository = InMemoryProfileRepository()

    with pytest.raises(ProfileNotFoundError):
        await repository.get_profile(uuid.uuid4())


@pytest.mark.asyncio
async def test_update_existing_profile() -> None:
    repository = InMemoryProfileRepository()
    profile = await repository.save_profile(build_profile())

    updated_profile = profile.model_copy(
        update={
            "constraints": profile.constraints.model_copy(
                update={"weekly_hours_available": 15}
            ),
        }
    )

    result = await repository.update_profile(updated_profile)
    retrieved_profile = await repository.get_profile(profile.id)

    assert result.constraints.weekly_hours_available == 15
    assert retrieved_profile.constraints.weekly_hours_available == 15


@pytest.mark.asyncio
async def test_update_missing_profile_raises_not_found() -> None:
    repository = InMemoryProfileRepository()
    profile = build_profile()

    with pytest.raises(ProfileNotFoundError):
        await repository.update_profile(profile)


@pytest.mark.asyncio
async def test_delete_existing_profile() -> None:
    repository = InMemoryProfileRepository()
    profile = await repository.save_profile(build_profile())

    await repository.delete_profile(profile.id)

    with pytest.raises(ProfileNotFoundError):
        await repository.get_profile(profile.id)


@pytest.mark.asyncio
async def test_delete_missing_profile_raises_not_found() -> None:
    repository = InMemoryProfileRepository()

    with pytest.raises(ProfileNotFoundError):
        await repository.delete_profile(uuid.uuid4())


def test_profile_repository_is_abstract() -> None:
    with pytest.raises(TypeError):
        ProfileRepository()


def test_repository_errors_share_base_class() -> None:
    assert issubclass(ProfileNotFoundError, ProfileRepositoryError)
    assert issubclass(ProfileConcurrencyError, ProfileRepositoryError)
