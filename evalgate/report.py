"""Report generation — text, JSON, and GitHub Step Summary output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .drift import DriftReport, RunSummary


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def text_summary(summary: RunSummary, drift: DriftReport | None = None) -> str:
    """Return a human-readable text report."""
    lines: list[str] = [
        f"Eval run : {summary.run_id}",
        f"Dataset  : {summary.dataset}",
        f"Model    : {summary.model}",
        f"Results  : {summary.passed}/{summary.total} passed ({_pct(summary.pass_rate)})",
    ]
    if drift is not None:
        direction = "+" if drift.pass_rate_delta >= 0 else ""
        lines.append(f"Drift    : {direction}{_pct(drift.pass_rate_delta)} vs baseline")
        if drift.regressed:
            lines.append("WARNING  : regression exceeds threshold — gate FAILED")
    return "\n".join(lines)


def github_step_summary(
    summary: RunSummary,
    drift: DriftReport | None = None,
) -> str:
    """Return GitHub-flavored Markdown for $GITHUB_STEP_SUMMARY."""
    status = "PASS" if (drift is None or not drift.regressed) else "FAIL"
    lines: list[str] = [
        f"## Eval Gate — {status}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Dataset | {summary.dataset} |",
        f"| Model | {summary.model} |",
        f"| Pass rate | {_pct(summary.pass_rate)} ({summary.passed}/{summary.total}) |",
    ]
    if drift is not None:
        direction = "+" if drift.pass_rate_delta >= 0 else ""
        lines.append(f"| Drift vs baseline | {direction}{_pct(drift.pass_rate_delta)} |")
    return "\n".join(lines)


def write_json_report(
    summary: RunSummary,
    drift: DriftReport | None,
    out_path: Path | str,
) -> None:
    """Write a machine-readable JSON report."""
    payload: dict[str, Any] = {"summary": summary.model_dump()}
    if drift is not None:
        payload["drift"] = drift.model_dump()
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
