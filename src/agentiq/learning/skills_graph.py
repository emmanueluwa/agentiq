from __future__ import annotations

from dataclasses import dataclass, field


class SkillGraphError(Exception):
    """Raised when the skill graph is invalid or cannot satisfy an operation."""


class DuplicateSkillError(SkillGraphError):
    pass


class UnknownSkillError(SkillGraphError):
    pass


class CircularDependencyError(SkillGraphError):
    pass


@dataclass(frozen=True, slots=True)
class SkillNode:
    skill_id: str
    name: str
    description: str
    domain: str
    difficulty: int
    estimated_effort_hours: int | None = None
    prerequisite_skill_ids: frozenset[str] = field(default_factory=frozenset)
    related_skill_ids: frozenset[str] = field(default_factory=frozenset)
    mastery_criteria: tuple[str, ...] = ()
    common_misconceptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.skill_id.strip():
            raise ValueError("skill_id must not be empty.")

        if not self.name.strip():
            raise ValueError("name must not be empty.")

        if not self.domain.strip():
            raise ValueError("domain must not be empty.")

        if not 1 <= self.difficulty <= 10:
            raise ValueError("difficulty must be between 1 and 10.")

        if self.estimated_effort_hours is not None and self.estimated_effort_hours < 0:
            raise ValueError("estimated_effort_hours cannot be negative.")

        if self.skill_id in self.prerequisite_skill_ids:
            raise ValueError("A skill cannot be its own prerequisite.")


class SkillGraph:
    """A directed, acyclic graph of learnable engineering skills.

    The graph is domain-agnostic. It can represent Python fundamentals,
    Rust ownership, distributed systems, security, product discovery,
    system design, career development, or any other learning domain.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillNode] = {}

    @property
    def skills(self) -> dict[str, SkillNode]:
        return dict(self._skills)

    def add_skill(self, skill: SkillNode) -> SkillNode:
        if skill.skill_id in self._skills:
            raise DuplicateSkillError(f"Skill already exists: {skill.skill_id}")

        self._validate_references(
            skill=skill,
            reference_ids=skill.prerequisite_skill_ids,
            reference_name="prerequisite",
        )

        self._skills[skill.skill_id] = skill
        return skill

    def get_skill(self, skill_id: str) -> SkillNode:
        skill = self._skills.get(skill_id)

        if skill is None:
            raise UnknownSkillError(f"Skill not found: {skill_id}")

        return skill

    def has_skill(self, skill_id: str) -> bool:
        return skill_id in self._skills

    def add_prerequisite(self, skill_id: str, prerequisite_skill_id: str) -> None:
        skill = self.get_skill(skill_id)
        prerequisite = self.get_skill(prerequisite_skill_id)

        if skill_id == prerequisite_skill_id:
            raise CircularDependencyError("A skill cannot depend on itself.")

        updated_prerequisites = frozenset(
            {*skill.prerequisite_skill_ids, prerequisite_skill_id}
        )

        updated_skill = SkillNode(
            skill_id=skill.skill_id,
            name=skill.name,
            description=skill.description,
            domain=skill.domain,
            difficulty=skill.difficulty,
            estimated_effort_hours=skill.estimated_effort_hours,
            prerequisite_skill_ids=updated_prerequisites,
            related_skill_ids=skill.related_skill_ids,
            mastery_criteria=skill.mastery_criteria,
            common_misconceptions=skill.common_misconceptions,
        )

        self._skills[skill_id] = updated_skill
        self._validate_acyclic(start_skill_id=skill_id)

    def get_prerequisites(self, skill_id: str) -> list[SkillNode]:
        skill = self.get_skill(skill_id)

        return [
            self.get_skill(prerequisite_id)
            for prerequisite_id in sorted(skill.prerequisite_skill_ids)
        ]

    def get_dependents(self, skill_id: str) -> list[SkillNode]:
        self.get_skill(skill_id)

        return [
            skill
            for skill in self._skills.values()
            if skill_id in skill.prerequisite_skill_ids
        ]

    def get_learnable_skills(
        self,
        mastered_skill_ids: set[str],
    ) -> list[SkillNode]:
        """Return skills whose prerequisites have been mastered."""
        learnable: list[SkillNode] = []

        for skill in self._skills.values():
            unmet_prerequisites = skill.prerequisite_skill_ids - mastered_skill_ids

            if not unmet_prerequisites:
                learnable.append(skill)

        return sorted(
            learnable,
            key=lambda skill: (
                skill.difficulty,
                skill.skill_id,
            ),
        )

    def get_learning_path(self, target_skill_id: str) -> list[SkillNode]:
        """Return a dependency-first path to the target skill."""
        target = self.get_skill(target_skill_id)

        visited: set[str] = set()
        path: list[SkillNode] = []

        self._depth_first_path(
            skill_id=target.skill_id,
            visited=visited,
            path=path,
        )

        return path

    def validate(self) -> None:
        """Validate that the complete graph is acyclic."""
        for skill_id in self._skills:
            self._validate_acyclic(start_skill_id=skill_id)

    def _validate_references(
        self,
        skill: SkillNode,
        reference_ids: frozenset[str],
        reference_name: str,
    ) -> None:
        missing = reference_ids - set(self._skills)

        if missing:
            raise UnknownSkillError(
                f"{reference_name.capitalize()} skills not found: "
                f"{', '.join(sorted(missing))}"
            )

    def _validate_acyclic(self, start_skill_id: str) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        self._depth_first_cycle_check(
            skill_id=start_skill_id,
            visiting=visiting,
            visited=visited,
        )

    def _depth_first_cycle_check(
        self,
        skill_id: str,
        visiting: set[str],
        visited: set[str],
    ) -> None:
        if skill_id in visiting:
            raise CircularDependencyError(
                f"Circular prerequisite dependency detected at: {skill_id}"
            )

        if skill_id in visited:
            return

        visiting.add(skill_id)

        skill = self.get_skill(skill_id)

        for prerequisite_id in sorted(skill.prerequisite_skill_ids):
            self._depth_first_cycle_check(
                skill_id=prerequisite_id,
                visiting=visiting,
                visited=visited,
            )

        visiting.remove(skill_id)
        visited.add(skill_id)

    def _depth_first_path(
        self,
        skill_id: str,
        visited: set[str],
        path: list[SkillNode],
    ) -> None:
        if skill_id in visited:
            return

        visited.add(skill_id)

        skill = self.get_skill(skill_id)

        for prerequisite_id in sorted(skill.prerequisite_skill_ids):
            self._depth_first_path(
                skill_id=prerequisite_id,
                visited=visited,
                path=path,
            )

        path.append(skill)
