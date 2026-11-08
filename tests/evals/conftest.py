from __future__ import annotations

from tests.evals.report import report_collector
from tests.evals.grader import write_eval_report


def pytest_sessionfinish(session, exitstatus):
    if report_collector.results:
        write_eval_report(report_collector.results)
