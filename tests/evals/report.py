from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from tests.evals.grader import write_eval_report


class EvalReportCollector:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def add_quality_result(
        self,
        case: dict,
        assistant_response: str,
        done_data: dict,
        quality,
        passed: bool,
    ) -> None:
        self.results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "response": assistant_response,
                "done": done_data,
                "judge": asdict(quality),
                "passed": passed,
            }
        )

    def add_safety_result(
        self,
        case: dict,
        assistant_response: str,
        done_data: dict,
        safety,
        passed: bool,
    ) -> None:
        self.results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "input": case["input"],
                "response": assistant_response,
                "done": done_data,
                "safety": asdict(safety),
                "passed": passed,
            }
        )

    def write(self) -> None:
        if self.results:
            write_eval_report(self.results)


report_collector = EvalReportCollector()
