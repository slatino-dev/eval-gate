"""Tests for report.py rendering correctness.

Covers the delta sign rendering fix: positive deltas must render as '+0.140'
not '++0.140', and negative deltas must render as '-0.140' not '--0.140'.
"""

from __future__ import annotations

from evalgate.drift import DriftResult, RunSummary, ScorerDrift
from evalgate.report import github_step_summary, text_summary


def _make_scorer_drift(
    *,
    scorer: str = "exact",
    baseline_mean: float = 0.80,
    candidate_mean: float,
    regressed: bool = False,
    improved: bool = False,
) -> ScorerDrift:
    delta = candidate_mean - baseline_mean
    return ScorerDrift(
        scorer=scorer,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        delta=delta,
        ci_low=0.75,
        ci_high=0.85,
        regressed=regressed,
        improved=improved,
        significant=regressed or improved,
    )


def _make_summary() -> RunSummary:
    return RunSummary(
        run_id="abc12345",
        dataset="test-dataset",
        model="test-model",
        total=10,
        passed=8,
        failed=2,
        pass_rate=0.8,
    )


def _make_drift_result(scorer_drift: ScorerDrift) -> DriftResult:
    return DriftResult(
        scorers=[scorer_drift],
        regressed=scorer_drift.regressed,
        missing_scorers=[],
        new_scorers=[],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Delta sign rendering — the core regression the fix addresses
# ──────────────────────────────────────────────────────────────────────────────


def test_positive_delta_renders_with_single_plus() -> None:
    """Positive delta should render as '+0.140', not '++0.140'."""
    drift = _make_scorer_drift(candidate_mean=0.94, improved=True)
    result = _make_drift_result(drift)
    summary = _make_summary()

    text = text_summary(summary, result)
    assert "+0.140" in text, f"Expected '+0.140' in output, got:\n{text}"
    assert "++0.140" not in text, f"Double-plus found in output:\n{text}"


def test_negative_delta_renders_with_single_minus() -> None:
    """Negative delta should render as '-0.200', not '--0.200'."""
    drift = _make_scorer_drift(candidate_mean=0.60, regressed=True)
    result = _make_drift_result(drift)
    summary = _make_summary()

    text = text_summary(summary, result)
    assert "-0.200" in text, f"Expected '-0.200' in output, got:\n{text}"
    assert "--0.200" not in text, f"Double-minus found in output:\n{text}"


def test_positive_delta_in_github_step_summary() -> None:
    """The markdown table in github_step_summary also renders a single '+'."""
    drift = _make_scorer_drift(candidate_mean=0.94, improved=True)
    result = _make_drift_result(drift)
    summary = _make_summary()

    md = github_step_summary(summary, result)
    assert "+0.140" in md, f"Expected '+0.140' in markdown, got:\n{md}"
    assert "++0.140" not in md, f"Double-plus found in markdown:\n{md}"


def test_zero_delta_renders_with_plus() -> None:
    """Zero delta should render as '+0.000' (no regression, no improvement)."""
    drift = _make_scorer_drift(candidate_mean=0.80)
    result = _make_drift_result(drift)
    summary = _make_summary()

    text = text_summary(summary, result)
    assert "+0.000" in text, f"Expected '+0.000' in output, got:\n{text}"


# ──────────────────────────────────────────────────────────────────────────────
# Verdict strings
# ──────────────────────────────────────────────────────────────────────────────


def test_regressed_verdict_in_text_summary() -> None:
    drift = _make_scorer_drift(candidate_mean=0.60, regressed=True)
    result = _make_drift_result(drift)
    text = text_summary(_make_summary(), result)
    assert "REGRESSED" in text


def test_improved_verdict_in_text_summary() -> None:
    drift = _make_scorer_drift(candidate_mean=0.94, improved=True)
    result = _make_drift_result(drift)
    text = text_summary(_make_summary(), result)
    assert "improved" in text


def test_gate_failed_label_on_regression() -> None:
    drift = _make_scorer_drift(candidate_mean=0.60, regressed=True)
    result = _make_drift_result(drift)
    text = text_summary(_make_summary(), result)
    assert "FAILED" in text


def test_gate_passed_label_on_noise() -> None:
    drift = _make_scorer_drift(candidate_mean=0.80)
    result = _make_drift_result(drift)
    text = text_summary(_make_summary(), result)
    assert "PASSED" in text
