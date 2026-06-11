"""CLI entry-point for evalgate.

Commands
--------
evalgate run <dataset>
    Run an eval suite against an OpenAI-compatible endpoint (or a command
    adapter when --command is given).  With ``--k N`` the suite is repeated
    N times and results are compared against a committed baseline using the
    noise-floor statistical gate (:func:`~evalgate.drift.compare_to_baseline`).
    Without ``--k`` (or ``--k 1``) the legacy point-estimate gate is used.

evalgate baseline <dataset>
    Run the suite k times (``--k``, default 5), compute the noise-floor
    baseline, and write it to ``--out`` (default ``baseline.json``).

evalgate summary <report>
    Pretty-print a previously saved JSON report.

All user-supplied inputs that reach a subprocess are passed as discrete list
elements to ``execFileSync`` / ``subprocess.run`` — no shell is invoked.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import typer

from .adapters import ChatMessage, CommandAdapter, CompletionRequest, OpenAIAdapter
from .dataset import EvalCase, EvalDataset
from .drift import (
    Baseline,
    DriftResult,
    RunSummary,
    compare_to_baseline,
    detect_drift,
    summarize_run,
)
from .report import github_step_summary, text_summary, write_json_report
from .scorers import SCORER_REGISTRY, Scorer, get_scorer

app = typer.Typer(
    name="evalgate",
    help="Regression-eval CI gate for LLM/agent pipelines.",
    add_completion=False,
)

_SCORER_NAMES = sorted(SCORER_REGISTRY)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _case_to_prompt(case: EvalCase) -> str:
    """Coerce a case input to the string sent to the system under test."""
    return case.input if isinstance(case.input, str) else json.dumps(case.input)


def _run_once(
    ds: EvalDataset,
    scorers: list[str],
    scorer_objs: Sequence[Scorer],
    get_actual: Callable[[str], str],
) -> dict[str, list[float]]:
    """Run the dataset once and return per-scorer raw scores (floats in [0,1]).

    Skipped results (e.g. judge without an endpoint) are excluded from the list
    so they do not inflate or deflate the mean — the caller must handle an empty
    list for a scorer if all cases were skipped.
    """
    raw: dict[str, list[float]] = {name: [] for name in scorers}
    for case in ds.cases:
        prompt = _case_to_prompt(case)
        try:
            actual: str = get_actual(prompt)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[WARN] case {case.id}: adapter error — {exc}", err=True)
            # Count the case as 0.0 for all scorers (the adapter failed).
            for name in scorers:
                raw[name].append(0.0)
            continue
        for name, scorer_obj in zip(scorers, scorer_objs):
            result = scorer_obj.score(actual, case.expected)
            if result.skipped:
                continue  # exclude skipped results from the mean
            raw[name].append(result.score)

    return raw


def _load_dataset(path: Path) -> EvalDataset:
    """Load a dataset from a YAML or JSONL file, dispatching on extension."""
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return EvalDataset.from_jsonl(path)
    return EvalDataset.from_yaml(path)


def _build_adapter_fn(
    base_url: str,
    model: str,
    api_key: str,
    command: str | None,
) -> Callable[[str], str]:
    """Return a callable ``(prompt: str) -> str`` for the chosen adapter."""
    if command:
        cmd_adapter = CommandAdapter(cmd=command)
        return cmd_adapter.run
    else:
        http_adapter = OpenAIAdapter(base_url=base_url, api_key=api_key)

        def _http(prompt: str) -> str:
            req = CompletionRequest(
                model=model,
                messages=[ChatMessage(role="user", content=prompt)],
            )
            return http_adapter.complete(req).content

        return _http


def _single_run_summary(
    ds: EvalDataset,
    scorers_list: list[str],
    scorer_objs: Sequence[Scorer],
    get_actual: Callable[[str], str],
    model: str,
) -> RunSummary:
    """Run once, compute aggregate pass/fail, and return a RunSummary.

    Pass/fail and scores come from the same generation: each case is called
    once, the response is scored by every scorer, and the per-case pass verdict
    is derived directly from those same scores.  A case passes when *any*
    non-skipped scorer gives score >= 0.5.
    """
    total = len(ds.cases)
    passed = 0
    per_scorer_scores: dict[str, list[float]] = {name: [] for name in scorers_list}

    for case in ds.cases:
        prompt = _case_to_prompt(case)
        try:
            actual: str = get_actual(prompt)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"[WARN] case {case.id}: adapter error — {exc}", err=True)
            # Adapter failed — count 0.0 for all scorers and mark as failed.
            for name in scorers_list:
                per_scorer_scores[name].append(0.0)
            continue

        case_passed = False
        for name, scorer_obj in zip(scorers_list, scorer_objs):
            result = scorer_obj.score(actual, case.expected)
            if result.skipped:
                continue
            per_scorer_scores[name].append(result.score)
            if result.score >= 0.5:
                case_passed = True
        if case_passed:
            passed += 1

    pass_rate = passed / total if total > 0 else 0.0
    scores: dict[str, float] = {}
    for name, values in per_scorer_scores.items():
        if values:
            scores[name] = sum(values) / len(values)

    return RunSummary(
        run_id=str(uuid.uuid4())[:8],
        dataset=ds.name,
        model=model,
        total=total,
        passed=passed,
        failed=total - passed,
        pass_rate=pass_rate,
        scores=scores,
    )


# ──────────────────────────────────────────────────────────────────────────────
# evalgate run
# ──────────────────────────────────────────────────────────────────────────────


@app.command()
def run(
    dataset: Annotated[Path, typer.Argument(help="Path to YAML or JSONL eval dataset")],
    base_url: Annotated[
        str, typer.Option(help="OpenAI-compatible API base URL")
    ] = "http://localhost:8000",
    model: Annotated[str, typer.Option(help="Model name to request")] = "default",
    api_key: Annotated[str, typer.Option(envvar="EVALGATE_API_KEY")] = "not-needed",
    command: Annotated[
        str | None,
        typer.Option(
            help=(
                "Spawn a subprocess instead of calling an API. "
                "Use {input} as placeholder for the case input. "
                "Example: --command='my-tool --query {input}'"
            )
        ),
    ] = None,
    k: Annotated[
        int,
        typer.Option(
            help=(
                "Number of repeated runs for the statistical noise-floor gate. "
                "k >= 3 activates compare_to_baseline(); k=1 uses the legacy "
                "point-estimate gate (requires --baseline to be a RunSummary)."
            )
        ),
    ] = 1,
    scorer: Annotated[
        str,
        typer.Option(
            help=(
                f"Comma-separated list of scorers to use. "
                f"Available: {', '.join(_SCORER_NAMES)}"
            )
        ),
    ] = "contains",
    baseline: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Baseline file. For k >= 3 this must be a Baseline JSON produced by "
                "'evalgate baseline'. For k=1 it must be a RunSummary JSON."
            )
        ),
    ] = None,
    threshold: Annotated[
        float, typer.Option(help="Pass-rate drop threshold for the k=1 gate (0.0–1.0)")
    ] = 0.05,
    out: Annotated[Path | None, typer.Option(help="Write JSON report to this path")] = None,
    github_summary: Annotated[
        bool,
        typer.Option("--github-summary", help="Also emit GitHub Step Summary markdown to stdout"),
    ] = False,
) -> None:
    """Run an eval suite and optionally compare against a baseline."""
    ds = _load_dataset(dataset)

    # Parse comma-separated scorer list.
    scorer_names = [s.strip() for s in scorer.split(",") if s.strip()]
    if not scorer_names:
        typer.echo("ERROR: --scorer cannot be empty", err=True)
        raise typer.Exit(code=2)
    try:
        scorer_objs = [get_scorer(name) for name in scorer_names]
    except ValueError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    get_actual = _build_adapter_fn(base_url, model, api_key, command)

    # ── Statistical path (k >= 3) ─────────────────────────────────────────────
    if k >= 3:
        # Collect k repeat x n cases per scorer.
        repeat_raw: dict[str, list[list[float]]] = {name: [] for name in scorer_names}
        for repeat_idx in range(k):
            typer.echo(f"[run {repeat_idx + 1}/{k}] …", err=True)
            raw = _run_once(ds, scorer_names, scorer_objs, get_actual)
            for name in scorer_names:
                if not raw[name]:
                    typer.echo(
                        f"[WARN] scorer '{name}' repeat {repeat_idx + 1}: "
                        "all cases skipped — this repeat is excluded from the mean",
                        err=True,
                    )
                else:
                    repeat_raw[name].append(raw[name])

        # Candidate means for compare_to_baseline: average the per-repeat means
        # (consistent with how summarize_run defines the baseline mean).
        candidate_means: dict[str, float] = {}
        for name, repeats in repeat_raw.items():
            if not repeats:
                typer.echo(
                    f"[WARN] scorer '{name}': all repeats were skipped — "
                    "excluding from comparison",
                    err=True,
                )
                continue
            repeat_means = [sum(rep) / len(rep) for rep in repeats]
            candidate_means[name] = sum(repeat_means) / len(repeat_means)

        # Also emit a RunSummary for the report (using last repeat's counts).
        total = len(ds.cases)
        # simple aggregate for display
        all_combined = [s for rep in (repeat_raw.get(scorer_names[0], [[]])) for s in rep]
        passed_est = sum(1 for s in all_combined if s >= 0.5)
        pass_rate = passed_est / len(all_combined) if all_combined else 0.0
        summary = RunSummary(
            run_id=str(uuid.uuid4())[:8],
            dataset=ds.name,
            model=model if not command else "(command)",
            total=total,
            passed=passed_est,
            failed=len(all_combined) - passed_est,
            pass_rate=pass_rate,
            scores=candidate_means,
        )

        drift_result: DriftResult | None = None
        if baseline is not None:
            try:
                bl = Baseline.load(baseline)
            except Exception as exc:  # noqa: BLE001
                typer.echo(f"ERROR loading baseline: {exc}", err=True)
                raise typer.Exit(code=2) from exc
            drift_result = compare_to_baseline(bl, candidate_means)

        typer.echo(text_summary(summary, drift_result))
        if github_summary:
            typer.echo(github_step_summary(summary, drift_result))

        if out is not None:
            write_json_report(summary, drift_result, out)
            typer.echo(f"Report written to {out}")

        if drift_result is not None and drift_result.regressed:
            raise typer.Exit(code=1)
        return

    # ── k=2 is not allowed (below MIN_REPEATS for the t-interval) ─────────────
    if k == 2:
        typer.echo(
            "ERROR: k=2 is not supported. Use k=1 (point-estimate gate) "
            "or k >= 3 (statistical gate).",
            err=True,
        )
        raise typer.Exit(code=2)

    # ── Legacy single-run path (k=1) ──────────────────────────────────────────
    summary = _single_run_summary(ds, scorer_names, scorer_objs, get_actual, model)

    drift_report = None
    if baseline is not None:
        try:
            base_summary = RunSummary.from_file(baseline)
            drift_report = detect_drift(base_summary, summary, threshold=threshold)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"ERROR loading baseline: {exc}", err=True)
            raise typer.Exit(code=2) from exc

    typer.echo(text_summary(summary, drift_report))
    if github_summary:
        typer.echo(github_step_summary(summary, drift_report))

    if out is not None:
        write_json_report(summary, drift_report, out)
        typer.echo(f"Report written to {out}")

    if drift_report is not None and drift_report.regressed:
        raise typer.Exit(code=1)


# ──────────────────────────────────────────────────────────────────────────────
# evalgate baseline
# ──────────────────────────────────────────────────────────────────────────────


@app.command()
def baseline(
    dataset: Annotated[Path, typer.Argument(help="Path to YAML or JSONL eval dataset")],
    base_url: Annotated[
        str, typer.Option(help="OpenAI-compatible API base URL")
    ] = "http://localhost:8000",
    model: Annotated[str, typer.Option(help="Model name to request")] = "default",
    api_key: Annotated[str, typer.Option(envvar="EVALGATE_API_KEY")] = "not-needed",
    command: Annotated[
        str | None,
        typer.Option(help="Spawn a subprocess instead of calling an API. Use {input} as placeholder."),
    ] = None,
    k: Annotated[
        int,
        typer.Option(help="Number of repeated runs (>= 3) for the noise-floor baseline"),
    ] = 5,
    scorer: Annotated[
        str,
        typer.Option(
            help=f"Comma-separated scorer names. Available: {', '.join(_SCORER_NAMES)}"
        ),
    ] = "contains",
    out: Annotated[
        Path, typer.Option(help="Write Baseline JSON to this path")
    ] = Path("baseline.json"),
    meta_model: Annotated[
        str | None,
        typer.Option("--meta-model", help="Store model name in baseline metadata"),
    ] = None,
) -> None:
    """Run the suite k times and write a statistical noise-floor baseline."""
    if k < 3:
        typer.echo("ERROR: --k must be >= 3 for a meaningful baseline", err=True)
        raise typer.Exit(code=2)

    ds = _load_dataset(dataset)

    scorer_names = [s.strip() for s in scorer.split(",") if s.strip()]
    try:
        scorer_objs = [get_scorer(name) for name in scorer_names]
    except ValueError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    get_actual = _build_adapter_fn(base_url, model, api_key, command)

    repeat_raw: dict[str, list[list[float]]] = {name: [] for name in scorer_names}
    for repeat_idx in range(k):
        typer.echo(f"[baseline run {repeat_idx + 1}/{k}] …", err=True)
        raw = _run_once(ds, scorer_names, scorer_objs, get_actual)
        for name in scorer_names:
            if not raw[name]:
                typer.echo(
                    f"[WARN] scorer '{name}' repeat {repeat_idx + 1}: "
                    "all cases skipped — this repeat is excluded from the baseline",
                    err=True,
                )
            else:
                repeat_raw[name].append(raw[name])

    # Error if any scorer has too few valid repeats to build a t-interval.
    for name, repeats in repeat_raw.items():
        if len(repeats) < 3:
            typer.echo(
                f"ERROR: scorer '{name}' has only {len(repeats)} non-skipped "
                "repeat(s) — need >= 3 for a meaningful baseline. "
                "Check that the scorer endpoint is reachable.",
                err=True,
            )
            raise typer.Exit(code=2)

    meta: dict[str, str] = {"dataset": ds.name}
    if meta_model:
        meta["model"] = meta_model
    elif not command:
        meta["model"] = model

    bl = summarize_run(repeat_raw, meta=meta)
    bl.save(out)
    typer.echo(f"Baseline written to {out}")
    for name, stats in bl.scorers.items():
        typer.echo(
            f"  {name}: mean={stats.mean:.3f}  "
            f"CI=[{stats.ci_low:.3f}, {stats.ci_high:.3f}]  k={stats.k}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# evalgate summary
# ──────────────────────────────────────────────────────────────────────────────


@app.command()
def summary(
    report: Annotated[Path, typer.Argument(help="JSON report produced by 'run'")],
    github: Annotated[bool, typer.Option(help="Output GitHub Step Summary markdown")] = False,
) -> None:
    """Pretty-print a saved report."""
    with open(report) as fh:
        data = json.load(fh)
    s = RunSummary.model_validate(data["summary"])

    drift: DriftResult | None = None
    if "drift" in data:
        drift = DriftResult.model_validate(data["drift"])

    typer.echo(text_summary(s, drift))
    if github:
        typer.echo(github_step_summary(s, drift))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
