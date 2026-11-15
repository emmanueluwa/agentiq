from __future__ import annotations

import pytest

from agentiq.learning.skill_graph import (
    CircularDependencyError,
    DuplicateSkillError,
    SkillGraph,
    SkillGraphError,
    SkillNode,
    UnknownSkillError,
)


def build_skill(
    skill_id: str = "python_basics",
    prerequisites: frozenset[str] | None = None,
) -> SkillNode:
    return SkillNode(
        skill_id=skill_id,
        name=skill_id.replace("_", " ").title(),
        description=f"Learn {skill_id.replace('_', ' ')}.",
        domain="programming",
        difficulty=2,
        prerequisite_skill_ids=prerequisites or frozenset(),
    )


def build_graph() -> SkillGraph:
    graph = SkillGraph()
    graph.add_skill(build_skill("python_basics"))
    graph.add_skill(
        build_skill(
            "python_functions",
            prerequisites=frozenset({"python_basics"}),
        )
    )
    graph.add_skill(
        build_skill(
            "python_testing",
            prerequisites=frozenset({"python_functions"}),
        )
    )
    return graph


def test_add_skill_stores_skill() -> None:
    graph = SkillGraph()
    skill = build_skill()

    stored_skill = graph.add_skill(skill)

    assert stored_skill.skill_id == "python_basics"
    assert graph.has_skill("python_basics")
    assert graph.get_skill("python_basics") == skill


def test_add_duplicate_skill_raises_error() -> None:
    graph = SkillGraph()
    graph.add_skill(build_skill())

    with pytest.raises(DuplicateSkillError):
        graph.add_skill(build_skill())


def test_add_skill_with_unknown_prerequisite_raises_error() -> None:
    graph = SkillGraph()

    with pytest.raises(UnknownSkillError):
        graph.add_skill(
            build_skill(
                skill_id="python_functions",
                prerequisites=frozenset({"missing_skill"}),
            )
        )


def test_add_skill_rejects_self_dependency() -> None:
    graph = SkillGraph()

    with pytest.raises(ValueError):
        graph.add_skill(
            build_skill(
                skill_id="python_basics",
                prerequisites=frozenset({"python_basics"}),
            )
        )


def test_add_prerequisite_updates_skill() -> None:
    graph = build_graph()

    graph.add_skill(build_skill("python_async"))
    graph.add_prerequisite("python_async", "python_basics")

    skill = graph.get_skill("python_async")

    assert "python_basics" in skill.prerequisite_skill_ids


def test_add_self_prerequisite_raises_error() -> None:
    graph = build_graph()

    with pytest.raises(CircularDependencyError):
        graph.add_prerequisite("python_basics", "python_basics")


def test_circular_prerequisites_raise_error() -> None:
    graph = SkillGraph()
    graph.add_skill(build_skill("skill_a"))
    graph.add_skill(build_skill("skill_b"))
    graph.add_skill(build_skill("skill_c"))

    graph.add_prerequisite("skill_b", "skill_a")
    graph.add_prerequisite("skill_c", "skill_b")

    with pytest.raises(CircularDependencyError):
        graph.add_prerequisite("skill_a", "skill_c")


def test_get_unknown_skill_raises_error() -> None:
    graph = SkillGraph()

    with pytest.raises(UnknownSkillError):
        graph.get_skill("missing_skill")


def test_get_prerequisites_returns_dependency_order() -> None:
    graph = build_graph()

    prerequisites = graph.get_prerequisites("python_testing")

    assert [skill.skill_id for skill in prerequisites] == ["python_functions"]


def test_get_dependents_returns_skills_requiring_skill() -> None:
    graph = build_graph()

    dependents = graph.get_dependents("python_basics")

    assert [skill.skill_id for skill in dependents] == ["python_functions"]


def test_get_learning_path_returns_dependency_first_order() -> None:
    graph = build_graph()

    path = graph.get_learning_path("python_testing")

    assert [skill.skill_id for skill in path] == [
        "python_basics",
        "python_functions",
        "python_testing",
    ]


def test_get_learnable_skills_excludes_unmet_prerequisites() -> None:
    graph = build_graph()

    learnable = graph.get_learnable_skills(mastered_skill_ids=set())

    assert [skill.skill_id for skill in learnable] == ["python_basics"]

    learnable = graph.get_learnable_skills(mastered_skill_ids={"python_basics"})

    assert [skill.skill_id for skill in learnable] == [
        "python_basics",
        "python_functions",
    ]

    learnable = graph.get_learnable_skills(
        mastered_skill_ids={"python_basics", "python_functions"}
    )

    assert [skill.skill_id for skill in learnable] == [
        "python_basics",
        "python_functions",
        "python_testing",
    ]


def test_add_prerequisite_prevents_cycle_immediately() -> None:
    graph = SkillGraph()
    graph.add_skill(build_skill("skill_a"))
    graph.add_skill(build_skill("skill_b"))
    graph.add_skill(build_skill("skill_c"))

    graph.add_prerequisite("skill_b", "skill_a")
    graph.add_prerequisite("skill_c", "skill_b")

    with pytest.raises(CircularDependencyError):
        graph.add_prerequisite("skill_a", "skill_c")


def test_validate_passes_for_acyclic_graph() -> None:
    graph = build_graph()

    graph.validate()


def test_skill_graph_error_is_base_exception() -> None:
    assert issubclass(DuplicateSkillError, SkillGraphError)
    assert issubclass(UnknownSkillError, SkillGraphError)
    assert issubclass(CircularDependencyError, SkillGraphError)
