class LearningError(Exception):
    """Base exception for learning-domain failures."""


class ProfileNotFoundError(LearningError):
    """Raised when a learning profile cannot be found."""


class ProfileServiceError(LearningError):
    """Raised when a profile operation cannot be completed."""


class DiagnosticServiceError(LearningError):
    """Raised when a diagnostic operation cannot be completed."""
