"""Report generation — text, JSON, and GitHub Step Summary output.

Two drift types are supported:

- :class:`~evalgate.drift.DriftReport` — legacy single-run point-estimate gate
  produced by :func:`~evalgate.drift.detect_drift`.
- :class:`~evalgate.drift.DriftResult` — statistical per-scorer comparison
  produced by :func:`~evalgate.drift.compare_to_baseline`. Renders a markdown
  table with score, delta, significance, and regression flag per scorer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .drift import DriftReport, DriftResult, RunSummary


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _sign(value: float) -> str:
    return "+" if value >= 0 else ""


# ──────────────────────────────────────────────────────────────────────────────
# Per-scorer markdown table (DriftResult)
# ──────────────────────────────────────────────────────────────────────────────


def _scorer_table(drift_result: DriftResult) -> str:
    """Render a markdown table of per-scorer scores, deltas, and significance."""
    rows = [
        "| Scorer | Score | Delta | CI | Verdict |",
        "|--------|-------|-------|----|---------|",
    ]
    for d in drift_result.scorers:
        delta_str = f"{d.delta:+.3f}"
        ci_str = f"[{d.ci_low:.3f}, {d.ci_high:.3f}]"
        if d.regressed:
            verdict = "REGRESSED"
        elif d.improved:
            verdict = "improved"
        elif d.significant:
            verdict = "significant"
        else:
            verdict = "ok (noise)"
        rows.append(
            f"| {d.scorer} | {d.candidate_mean:.3f} | {delta_str} | {ci_str} | {verdict} |"
        )
    if drift_result.missing_scorers:
        rows.append("")
        rows.append(
            "_Scorers in baseline but absent from candidate: "
            + ", ".join(drift_result.missing_scorers)
            + "_"
        )
    if drift_result.new_scorers:
        rows.append("")
        rows.append(
            "_New scorers not in baseline: "
            + ", ".join(drift_result.new_scorers)
            + "_"
        )
    return "\n".join(rows)


def _scorer_table_text(drift_result: DriftResult) -> str:
    """Plain-text equivalent of the per-scorer table."""
    lines: list[str] = []
    for d in drift_result.scorers:
        delta_str = f"{d.delta:+.3f}"
        ci_str = f"[{d.ci_low:.3f}, {d.ci_high:.3f}]"
        if d.regressed:
            verdict = "REGRESSED"
        elif d.improved:
            verdict = "improved"
        elif d.significant:
            verdict = "significant"
        else:
            verdict = "ok"
        lines.append(
            f"  {d.scorer:20s}  score={d.candidate_mean:.3f}  "
            f"delta={delta_str}  CI={ci_str}  {verdict}"
        )
    if drift_result.missing_scorers:
        lines.append(f"  Missing scorers: {', '.join(drift_result.missing_scorers)}")
    if drift_result.new_scorers:
        lines.append(f"  New scorers: {', '.join(drift_result.new_scorers)}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Human-readable text summary
# ──────────────────────────────────────────────────────────────────────────────


def text_summary(
    summary: RunSummary,
    drift: DriftReport | DriftResult | None = None,
) -> str:
    """Return a human-readable text report."""
    lines: list[str] = [
        f"Eval run : {summary.run_id}",
        f"Dataset  : {summary.dataset}",
        f"Model    : {summary.model}",
        f"Results  : {summary.passed}/{summary.total} passed ({_pct(summary.pass_rate)})",
    ]
    if isinstance(drift, DriftResult):
        status = "FAILED (regression)" if drift.regressed else "PASSED"
        lines.append(f"Gate     : {status}")
        lines.append("Scorers  :")
        lines.append(_scorer_table_text(drift))
    elif isinstance(drift, DriftReport):
        direction = _sign(drift.pass_rate_delta)
        lines.append(f"Drift    : {direction}{_pct(drift.pass_rate_delta)} vs baseline")
        if drift.regressed:
            lines.append("WARNING  : regression exceeds threshold — gate FAILED")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# GitHub Step Summary markdown
# ──────────────────────────────────────────────────────────────────────────────


def github_step_summary(
    summary: RunSummary,
    drift: DriftReport | DriftResult | None = None,
) -> str:
    """Return GitHub-flavored Markdown for $GITHUB_STEP_SUMMARY."""
    if isinstance(drift, DriftResult):
        status = "FAIL" if drift.regressed else "PASS"
    elif isinstance(drift, DriftReport):
        status = "FAIL" if drift.regressed else "PASS"
    else:
        status = "PASS"

    lines: list[str] = [
        f"## Eval Gate — {status}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Dataset | {summary.dataset} |",
        f"| Model | {summary.model} |",
        f"| Pass rate | {_pct(summary.pass_rate)} ({summary.passed}/{summary.total}) |",
    ]

    if isinstance(drift, DriftResult):
        lines.append("")
        lines.append("### Per-scorer drift")
        lines.append("")
        lines.append(_scorer_table(drift))
    elif isinstance(drift, DriftReport):
        direction = _sign(drift.pass_rate_delta)
        lines.append(f"| Drift vs baseline | {direction}{_pct(drift.pass_rate_delta)} |")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Machine-readable JSON report
# ──────────────────────────────────────────────────────────────────────────────


def write_json_report(
    summary: RunSummary,
    drift: DriftReport | DriftResult | None,
    out_path: Path | str,
) -> None:
    """Write a machine-readable JSON report.

    The output always has a ``"summary"`` key.  When a drift result is
    present it is stored under ``"drift"`` (statistical path) or
    ``"drift_report"`` (legacy point-estimate path) so downstream tooling
    can tell them apart.
    """
    payload: dict[str, Any] = {"summary": summary.model_dump()}
    if isinstance(drift, DriftResult):
        payload["drift"] = drift.model_dump()
    elif isinstance(drift, DriftReport):
        payload["drift_report"] = drift.model_dump()
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
