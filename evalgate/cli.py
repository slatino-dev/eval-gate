"""CLI entry-point for evalgate."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import typer

from .adapters import ChatMessage, CompletionRequest, OpenAIAdapter
from .dataset import EvalDataset
from .drift import RunSummary, detect_drift
from .report import github_step_summary, text_summary, write_json_report
from .scorers import ContainsScorer, ExactMatchScorer

app = typer.Typer(
    name="evalgate",
    help="Regression-eval CI gate for LLM/agent pipelines.",
    add_completion=False,
)


@app.command()
def run(
    dataset: Annotated[Path, typer.Argument(help="Path to YAML eval dataset")],
    base_url: Annotated[str, typer.Option(help="OpenAI-compatible API base URL")] = "http://localhost:8000",
    model: Annotated[str, typer.Option(help="Model name to request")] = "default",
    api_key: Annotated[str, typer.Option(envvar="EVALGATE_API_KEY")] = "not-needed",
    baseline: Annotated[Path | None, typer.Option(help="JSON baseline from a previous run")] = None,
    threshold: Annotated[float, typer.Option(help="Max allowed pass-rate drop")] = 0.05,
    out: Annotated[Path | None, typer.Option(help="Write JSON report to this path")] = None,
    scorer: Annotated[str, typer.Option(help="Scorer: exact | contains")] = "contains",
) -> None:
    """Run an eval suite and optionally compare against a baseline."""
    ds = EvalDataset.from_yaml(dataset)
    adapter = OpenAIAdapter(base_url=base_url, api_key=api_key)
    scorer_obj = ExactMatchScorer() if scorer == "exact" else ContainsScorer()

    passed = 0
    for case in ds.cases:
        req = CompletionRequest(
            model=model,
            messages=[ChatMessage(role="user", content=case.prompt)],
        )
        try:
            resp = adapter.complete(req)
            result = scorer_obj.score(resp.content, case.expected)
            if result.passed:
                passed += 1
        except Exception as exc:
            typer.echo(f"[WARN] Case {case.id} errored: {exc}", err=True)

    total = len(ds.cases)
    pass_rate = passed / total if total > 0 else 0.0
    run_id = str(uuid.uuid4())[:8]

    summary = RunSummary(
        run_id=run_id,
        dataset=ds.name,
        model=model,
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=pass_rate,
    )

    drift_report = None
    if baseline is not None:
        base_summary = RunSummary.from_file(baseline)
        drift_report = detect_drift(base_summary, summary, threshold=threshold)

    typer.echo(text_summary(summary, drift_report))

    if out is not None:
        write_json_report(summary, drift_report, out)
        typer.echo(f"Report written to {out}")

    if drift_report is not None and drift_report.regressed:
        raise typer.Exit(code=1)


@app.command()
def summary(
    report: Annotated[Path, typer.Argument(help="JSON report produced by 'run'")],
    github: Annotated[bool, typer.Option(help="Output GitHub Step Summary markdown")] = False,
) -> None:
    """Pretty-print a saved report."""
    with open(report) as fh:
        data = json.load(fh)
    s = RunSummary.model_validate(data["summary"])
    typer.echo(text_summary(s))
    if github:
        typer.echo(github_step_summary(s))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
