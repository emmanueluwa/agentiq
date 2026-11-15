from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from agentiq.learning.models import LearnerProfile


class ProfileRepositoryError(Exception):
    """Raised when a profile repository operation cannot be completed safely."""


class ProfileNotFoundError(ProfileRepositoryError):
    """Raised when a requested learner profile does not exist."""


class ProfileConcurrencyError(ProfileRepositoryError):
    """Raised when concurrent profile writes conflict."""


class ProfileRepository(ABC):
    """Persistence boundary for learner profiles."""

    @abstractmethod
    async def save_profile(
        self,
        profile: LearnerProfile,
    ) -> LearnerProfile:
        """Create or replace a learner profile."""

    @abstractmethod
    async def get_profile(
        self,
        profile_id: UUID,
    ) -> LearnerProfile:
        """Retrieve a learner profile by ID."""

    @abstractmethod
    async def update_profile(
        self,
        profile: LearnerProfile,
    ) -> LearnerProfile:
        """Replace an existing learner profile."""

    @abstractmethod
    async def delete_profile(
        self,
        profile_id: UUID,
    ) -> None:
        """Delete a learner profile by ID."""


class InMemoryProfileRepository(ProfileRepository):
    """In-memory implementation for local development and tests.

    This is not durable production persistence. It is isolated behind
    ProfileRepository so it can be replaced by PostgreSQL or another durable
    backend without changing LearningProfileService.
    """

    def __init__(self) -> None:
        self._profiles: dict[UUID, LearnerProfile] = {}

    async def save_profile(
        self,
        profile: LearnerProfile,
    ) -> LearnerProfile:
        self._profiles[profile.id] = profile
        return profile

    async def get_profile(
        self,
        profile_id: UUID,
    ) -> LearnerProfile:
        profile = self._profiles.get(profile_id)

        if profile is None:
            raise ProfileNotFoundError(f"Learner profile not found: {profile_id}")

        return profile

    async def update_profile(
        self,
        profile: LearnerProfile,
    ) -> LearnerProfile:
        if profile.id not in self._profiles:
            raise ProfileNotFoundError(f"Learner profile not found: {profile.id}")

        self._profiles[profile.id] = profile
        return profile

    async def delete_profile(
        self,
        profile_id: UUID,
    ) -> None:
        if profile_id not in self._profiles:
            raise ProfileNotFoundError(f"Learner profile not found: {profile_id}")

        del self._profiles[profile_id]
