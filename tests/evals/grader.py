from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import AsyncOpenAI

from dotenv import load_dotenv

load_dotenv()


@dataclass
class JudgeResult:
    actionable: bool
    grounded: bool
    commercially_useful: bool
    hallucinated: bool
    score: int
    reason: str


JUDGE_SYSTEM_PROMPT = """
You are a rigorous senior engineering mentor evaluator.

You grade responses from AgentIQ, an AI mentor that helps engineers build
high-impact software products, advance their careers, and create commercial value.

Grade only using the user question and the response.

Prioritize:
- Concrete, actionable guidance.
- Honest uncertainty instead of fabrication.
- User impact, product thinking, and commercial outcomes.
- Technical judgment appropriate for production systems.
- Clear reasoning and practical next steps.

Return valid JSON only, with exactly these fields:
{
  "actionable": true | false,
  "grounded": true | false,
  "commercially_useful": true | false,
  "hallucinated": true | false,
  "score": 0-10,
  "reason": "brief explanation"
}
"""


def build_judge_user_prompt(question: str, response: str) -> str:
    return f"""
User question:
{question}

AgentIQ response:
{response}

Evaluate the response against AgentIQ's product mission:
help engineers create user impact, ship production-quality systems, and build
commercially valuable products.

Return JSON only.
"""


def parse_judge_json(raw: str) -> JudgeResult:
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()

    data = json.loads(cleaned)

    return JudgeResult(
        actionable=bool(data["actionable"]),
        grounded=bool(data["grounded"]),
        commercially_useful=bool(data["commercially_useful"]),
        hallucinated=bool(data["hallucinated"]),
        score=int(data["score"]),
        reason=str(data["reason"]),
    )


async def judge_response(
    question: str,
    response: str,
    model: str = "grok-4-fast-non-reasoning",
) -> JudgeResult:
    client = AsyncOpenAI(
        api_key=os.environ["GROK_API_KEY"],
        base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
    )

    completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_judge_user_prompt(question, response),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content or ""

    return parse_judge_json(raw)


def passes_production_bar(result: JudgeResult) -> bool:
    return (
        result.actionable
        and result.grounded
        and result.commercially_useful
        and not result.hallucinated
        and result.score >= 7
    )


def write_eval_report(
    results: list[dict],
    output_dir: Path = Path("output"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "eval-results.json"
    summary_path = output_dir / "eval-summary.md"

    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    passed = sum(1 for result in results if result["passed"])
    total = len(results)
    pass_rate = passed / total if total else 0

    lines = [
        "# AgentIQ Live Eval Summary",
        "",
        f"- Total cases: {total}",
        f"- Passed: {passed}",
        f"- Failed: {total - passed}",
        f"- Pass rate: {pass_rate:.1%}",
        "",
        "| Case | Category | Score | Passed | Reason |",
        "|---|---|---:|---:|---|",
    ]

    for result in results:
        judge = result["judge"]
        lines.append(
            f"| `{result['id']}` | {result['category']} "
            f"| {judge['score']} | {'✅' if result['passed'] else '❌'} "
            f"| {judge['reason'].replace('|', '\\|')} |"
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")
