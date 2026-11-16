from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from agentiq.learning.models import LearnerProfile


class ProfileRepositoryError(Exception):
    """Raised when a profile repository operation cannot be completed safely."""


class ProfileNotFoundError(ProfileRepositoryError):
    """Raised when a requested learner profile does not exist."""


class ProfileConcurrencyError(ProfileRepositoryError):
    """Raised when a profile update conflicts with a newer persisted version."""


class ProfileRepository(ABC):
    """Persistence boundary for learner profiles."""

    @abstractmethod
    async def save_profile(
        self,
        profile: LearnerProfile,
    ) -> LearnerProfile:
        """Create a learner profile."""

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
        expected_version: int,
    ) -> LearnerProfile:
        """Update a profile only when its persisted version is unchanged."""

    @abstractmethod
    async def delete_profile(
        self,
        profile_id: UUID,
    ) -> None:
        """Delete a learner profile by ID."""


class InMemoryProfileRepository(ProfileRepository):
    """In-memory implementation for local development and tests."""

    def __init__(self) -> None:
        self._profiles: dict[UUID, LearnerProfile] = {}

    async def save_profile(
        self,
        profile: LearnerProfile,
    ) -> LearnerProfile:
        if profile.id in self._profiles:
            raise ProfileConcurrencyError(
                f"Learner profile already exists: {profile.id}"
            )

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
        expected_version: int,
    ) -> LearnerProfile:
        current_profile = self._profiles.get(profile.id)

        if current_profile is None:
            raise ProfileNotFoundError(f"Learner profile not found: {profile.id}")

        if current_profile.version != expected_version:
            raise ProfileConcurrencyError(
                "Learner profile was modified by another operation."
            )

        updated_profile = profile.model_copy(
            update={"version": current_profile.version + 1}
        )

        self._profiles[profile.id] = updated_profile
        return updated_profile

    async def delete_profile(
        self,
        profile_id: UUID,
    ) -> None:
        if profile_id not in self._profiles:
            raise ProfileNotFoundError(f"Learner profile not found: {profile_id}")

        del self._profiles[profile_id]
