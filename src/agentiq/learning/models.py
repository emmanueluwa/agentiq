from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExperienceLevel(StrEnum):
    BEGINNER = "beginner"
    EARLY_CAREER = "early_career"
    MID_LEVEL = "mid_level"
    SENIOR = "senior"
    RETURNING = "returning"
    UNKNOWN = "unknown"


class PrimaryGoal(StrEnum):
    ENGINEERING_MASTERY = "engineering_mastery"
    INTERVIEW_PREPARATION = "interview_preparation"
    PRODUCT_BUILDING = "product_building"
    CAREER_ADVANCEMENT = "career_advancement"
    FREELANCING = "freelancing"
    ENTREPRENEURSHIP = "entrepreneurship"
    GENERAL_IMPROVEMENT = "general_improvement"


class MasteryStatus(StrEnum):
    NOT_STARTED = "not_started"
    LEARNING = "learning"
    PRACTICING = "practicing"
    PROFICIENT = "proficient"
    NEEDS_REVIEW = "needs_review"


class EvidenceType(StrEnum):
    SELF_REPORT = "self_report"
    CONCEPT_CHECK = "concept_check"
    CODE_EXERCISE = "code_exercise"
    DEBUGGING_EXERCISE = "debugging_exercise"
    ARCHITECTURE_REVIEW = "architecture_review"
    PROJECT_ARTIFACT = "project_artifact"
    MENTOR_REVIEW = "mentor_review"
    EXTERNAL_CREDENTIAL = "external_credential"


class AssessmentStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    NEEDS_MORE_INFORMATION = "needs_more_information"
    FAILED_VALIDATION = "failed_validation"


class LearningDomain(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)


class LearnerGoal(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=1, max_length=4000)
    primary_goal: PrimaryGoal
    target_outcome: str = Field(min_length=1, max_length=2000)
    target_completion_date: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


class LearnerConstraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    weekly_hours_available: int | None = Field(default=None, ge=0, le=120)
    preferred_session_minutes: int = Field(default=45, ge=10, le=240)
    preferred_learning_format: str = Field(default="video_first")
    preferred_language: str | None = Field(default=None, max_length=50)
    timezone: str | None = Field(default=None, max_length=100)
    deadline: datetime | None = None


class LearnerPreferences(BaseModel):
    model_config = ConfigDict(frozen=True)

    explanation_style: str = Field(default="direct_and_practical")
    feedback_style: str = Field(default="direct")
    resource_depth: str = Field(default="curated")
    prefer_videos: bool = True
    prefer_projects: bool = True
    prefer_assessments: bool = True


class SkillState(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str = Field(min_length=1, max_length=200)
    topic: str = Field(min_length=1, max_length=200)
    self_reported_level: int = Field(default=0, ge=0, le=10)
    demonstrated_level: int = Field(default=0, ge=0, le=10)
    mastery_status: MasteryStatus = MasteryStatus.NOT_STARTED
    last_practiced_at: datetime | None = None
    next_review_at: datetime | None = None
    attempt_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    common_mistakes: tuple[str, ...] = ()
    prerequisite_skill_ids: tuple[str, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()


class LearnerProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    experience_level: ExperienceLevel = ExperienceLevel.UNKNOWN
    primary_goal: PrimaryGoal = PrimaryGoal.GENERAL_IMPROVEMENT
    goals: tuple[LearnerGoal, ...] = ()
    constraints: LearnerConstraint = Field(default_factory=LearnerConstraint)
    preferences: LearnerPreferences = Field(default_factory=LearnerPreferences)

    known_languages: tuple[str, ...] = ()
    domains_of_interest: tuple[str, ...] = ()
    target_role: str | None = None
    target_timeline: str | None = None

    skill_states: dict[str, SkillState] = Field(default_factory=dict)


class AssessmentRubricCriterion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=1000)
    weight: float = Field(default=1.0, ge=0, le=10)
    passing_standard: str = Field(min_length=1, max_length=1000)


class AssessmentRubric(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=100)
    topic: str = Field(min_length=1, max_length=200)
    criteria: tuple[AssessmentRubricCriterion, ...]
    minimum_score: int = Field(default=7, ge=0, le=10)
    requires_evidence: bool = True
    allows_partial_credit: bool = True


class DiagnosticBlueprint(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    topic: str = Field(min_length=1, max_length=200)
    learning_objective: str = Field(min_length=1, max_length=2000)
    prerequisite_skill_ids: tuple[str, ...] = ()
    target_skill_ids: tuple[str, ...] = ()
    difficulty: int = Field(default=1, ge=1, le=5)
    question_count: int = Field(default=3, ge=1, le=10)
    assessment_types: tuple[EvidenceType, ...] = (EvidenceType.CONCEPT_CHECK,)
    rubric: AssessmentRubric
    created_at: datetime = Field(default_factory=utc_now)


class DiagnosticQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    blueprint_id: UUID
    topic: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=4000)
    question_type: EvidenceType
    difficulty: int = Field(default=1, ge=1, le=5)
    expected_concepts: tuple[str, ...] = ()
    rubric_id: str = Field(min_length=1, max_length=100)
    created_at: datetime = Field(default_factory=utc_now)


class DiagnosticResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: UUID
    response: str = Field(min_length=1, max_length=20000)
    artifact_url: HttpUrl | None = None
    time_spent_seconds: int | None = Field(default=None, ge=0)


class AssessmentCriterionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    criterion_id: str
    score: int = Field(ge=0, le=10)
    passed: bool
    rationale: str = Field(min_length=1, max_length=2000)
    evidence_ids: tuple[UUID, ...] = ()


class AssessmentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    profile_id: UUID
    blueprint_id: UUID
    topic: str = Field(min_length=1, max_length=200)
    status: AssessmentStatus
    overall_score: int = Field(ge=0, le=10)
    passed: bool
    criterion_results: tuple[AssessmentCriterionResult, ...] = ()
    strengths: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    prerequisite_gaps: tuple[str, ...] = ()
    uncertainty: tuple[str, ...] = ()
    next_action: str = Field(min_length=1, max_length=2000)
    assessor_notes: str | None = Field(default=None, max_length=4000)
    created_at: datetime = Field(default_factory=utc_now)


class AssessmentEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    profile_id: UUID
    assessment_result_id: UUID
    topic: str = Field(min_length=1, max_length=200)
    evidence_type: EvidenceType
    created_at: datetime = Field(default_factory=utc_now)

    score: int | None = Field(default=None, ge=0, le=10)
    passed: bool | None = None
    observed_strengths: tuple[str, ...] = ()
    observed_gaps: tuple[str, ...] = ()
    notes: str | None = Field(default=None, max_length=4000)
    artifact_url: HttpUrl | None = None


class CreateProfileRequest(BaseModel):
    experience_level: ExperienceLevel = ExperienceLevel.UNKNOWN
    primary_goal: PrimaryGoal = PrimaryGoal.GENERAL_IMPROVEMENT
    goal_title: str = Field(min_length=1, max_length=300)
    goal_description: str = Field(min_length=1, max_length=4000)
    target_outcome: str = Field(min_length=1, max_length=2000)
    known_languages: list[str] = Field(default_factory=list)
    domains_of_interest: list[str] = Field(default_factory=list)
    weekly_hours_available: int | None = Field(default=None, ge=0, le=80)
    preferred_session_minutes: int = Field(default=45, ge=10, le=240)
    preferred_learning_format: str = "video_first"
    preferred_language: str | None = Field(default=None, max_length=50)
    target_role: str | None = Field(default=None, max_length=200)
    target_timeline: str | None = Field(default=None, max_length=200)


class UpdateProfileRequest(BaseModel):
    experience_level: ExperienceLevel | None = None
    primary_goal: PrimaryGoal | None = None
    constraints: LearnerConstraint | None = None
    preferences: LearnerPreferences | None = None
    known_languages: list[str] | None = None
    domains_of_interest: list[str] | None = None
    target_role: str | None = Field(default=None, max_length=200)
    target_timeline: str | None = Field(default=None, max_length=200)


class RecordAssessmentRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    evidence_type: EvidenceType
    score: int | None = Field(default=None, ge=0, le=10)
    passed: bool | None = None
    observed_strengths: list[str] = Field(default_factory=list)
    observed_gaps: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=4000)
    artifact_url: HttpUrl | None = None
