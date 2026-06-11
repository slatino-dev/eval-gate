"""Hand-checked tests for the noise-floor drift math.

The t critical values are asserted against standard t-tables, the confidence
intervals against by-hand computations, and the regression flags against
synthetic distributions where the right answer is known by construction.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from evalgate.drift import (
    Baseline,
    DriftError,
    compare_to_baseline,
    compute_stats,
    summarize_run,
    t_critical,
)

# ──────────────────────────────────────────────────────────────────────────────
# Student-t critical values vs. standard tables
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("confidence", "df", "expected"),
    [
        (0.95, 1, 12.7062),
        (0.95, 2, 4.302653),
        (0.95, 4, 2.776445),
        (0.95, 9, 2.262157),
        (0.99, 9, 3.249836),
        (0.90, 29, 1.699127),
    ],
)
def test_t_critical_matches_t_tables(confidence: float, df: int, expected: float) -> None:
    assert t_critical(confidence, df) == pytest.approx(expected, abs=1e-4)


def test_t_critical_rejects_bad_inputs() -> None:
    with pytest.raises(DriftError, match="confidence"):
        t_critical(1.0, 5)
    with pytest.raises(DriftError, match="degrees of freedom"):
        t_critical(0.95, 0)


# ──────────────────────────────────────────────────────────────────────────────
# compute_stats — CI by hand
# ──────────────────────────────────────────────────────────────────────────────


def test_compute_stats_three_repeats_by_hand() -> None:
    # repeats [0.8, 0.9, 1.0]: mean 0.9, sample std 0.1
    # half-width = t(0.95, df=2) * 0.1 / sqrt(3) = 4.302653 * 0.0577350 = 0.2484137
    stats = compute_stats("exact", [0.8, 0.9, 1.0])
    assert stats.k == 3
    assert stats.mean == pytest.approx(0.9)
    assert stats.std == pytest.approx(0.1)
    assert stats.ci_low == pytest.approx(0.9 - 0.2484137, abs=1e-5)
    assert stats.ci_high == 1.0  # 0.9 + 0.248 clamped to the score bound


def test_compute_stats_five_repeats_by_hand() -> None:
    # repeats [0.88, 0.90, 0.92, 0.89, 0.91]: mean 0.9
    # var = (0.0004 + 0 + 0.0004 + 0.0001 + 0.0001) / 4 = 2.5e-4 → std 0.0158114
    # half-width = t(0.95, df=4) * std / sqrt(5) = 2.776445 * 0.0158114 / 2.2360680
    #            = 0.0196325
    stats = compute_stats("f1", [0.88, 0.90, 0.92, 0.89, 0.91])
    assert stats.mean == pytest.approx(0.9)
    assert stats.std == pytest.approx(0.0158114, abs=1e-6)
    assert stats.ci_low == pytest.approx(0.8803675, abs=1e-4)
    assert stats.ci_high == pytest.approx(0.9196325, abs=1e-4)


def test_compute_stats_requires_three_repeats() -> None:
    with pytest.raises(DriftError, match="at least 3 repeats"):
        compute_stats("exact", [0.9, 0.91])


def test_compute_stats_rejects_out_of_range_scores() -> None:
    with pytest.raises(DriftError, match=r"\[0, 1\]"):
        compute_stats("exact", [0.9, 0.9, 1.2])


def test_zero_variance_collapses_interval_to_a_point() -> None:
    stats = compute_stats("exact", [0.9, 0.9, 0.9])
    assert stats.std == 0.0
    assert stats.ci_low == stats.ci_high == 0.9


# ──────────────────────────────────────────────────────────────────────────────
# compare_to_baseline — interval comparison, not point comparison
# ──────────────────────────────────────────────────────────────────────────────


def _baseline() -> Baseline:
    # CI hand-checked above: [0.8803675, 0.9196325] around mean 0.9
    return Baseline(scorers={"f1": compute_stats("f1", [0.88, 0.90, 0.92, 0.89, 0.91])})


def test_drop_inside_the_noise_floor_is_ignored() -> None:
    # 0.89 is below the baseline mean 0.9 but inside the CI → NOT a regression.
    # A point-estimate comparison would have flagged this.
    result = compare_to_baseline(_baseline(), {"f1": 0.89})
    drift = result.scorers[0]
    assert drift.delta == pytest.approx(-0.01)
    assert not drift.regressed and not drift.improved and not drift.significant
    assert not result.regressed


def test_drop_below_the_interval_is_a_regression() -> None:
    result = compare_to_baseline(_baseline(), {"f1": 0.85})
    drift = result.scorers[0]
    assert drift.regressed and drift.significant and not drift.improved
    assert drift.delta == pytest.approx(-0.05)
    assert result.regressed


def test_rise_above_the_interval_is_improvement_not_regression() -> None:
    result = compare_to_baseline(_baseline(), {"f1": 0.95})
    drift = result.scorers[0]
    assert drift.improved and drift.significant and not drift.regressed
    assert not result.regressed


def test_boundary_is_exclusive_just_inside_vs_just_outside() -> None:
    inside = compare_to_baseline(_baseline(), {"f1": 0.881}).scorers[0]
    outside = compare_to_baseline(_baseline(), {"f1": 0.879}).scorers[0]
    assert not inside.regressed
    assert outside.regressed


def test_zero_variance_baseline_flags_any_drop() -> None:
    baseline = Baseline(scorers={"exact": compute_stats("exact", [0.9, 0.9, 0.9])})
    assert compare_to_baseline(baseline, {"exact": 0.899}).regressed
    assert not compare_to_baseline(baseline, {"exact": 0.9}).regressed


def test_missing_and_new_scorers_are_reported() -> None:
    result = compare_to_baseline(_baseline(), {"bleu": 0.5})
    assert result.missing_scorers == ["f1"]
    assert result.new_scorers == ["bleu"]
    assert result.scorers == [] and not result.regressed


def test_candidate_mean_out_of_range_rejected() -> None:
    with pytest.raises(DriftError, match=r"\[0, 1\]"):
        compare_to_baseline(_baseline(), {"f1": 1.3})


# ──────────────────────────────────────────────────────────────────────────────
# summarize_run — repeats x cases → baseline
# ──────────────────────────────────────────────────────────────────────────────


def test_summarize_run_reduces_repeats_to_means() -> None:
    # repeat means: [0.75, 0.5, 1.0] → mean 0.75, std 0.25
    # half-width = 4.302653 * 0.25 / sqrt(3) = 0.6210342 → ci [0.1289658, 1.0]
    baseline = summarize_run({"exact": [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]]})
    stats = baseline.scorers["exact"]
    assert stats.mean == pytest.approx(0.75)
    assert stats.std == pytest.approx(0.25)
    assert stats.ci_low == pytest.approx(0.1289658, abs=1e-5)
    assert stats.ci_high == 1.0


def test_summarize_run_rejects_empty_repeat() -> None:
    with pytest.raises(DriftError, match="repeat 1 has no case scores"):
        summarize_run({"exact": [[1.0], [], [0.5]]})


def test_baseline_round_trips_through_json(tmp_path: Path) -> None:
    baseline = summarize_run({"exact": [[1, 0, 1], [1, 1, 1], [0, 1, 1]]}, meta={"model": "m1"})
    path = tmp_path / "baseline.json"
    baseline.save(path)
    assert Baseline.load(path) == baseline


def test_baseline_load_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(DriftError, match="not found"):
        Baseline.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DriftError, match="invalid JSON"):
        Baseline.load(bad)
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"scorers": "nope"}', encoding="utf-8")
    with pytest.raises(DriftError, match="invalid baseline"):
        Baseline.load(wrong)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic end-to-end: known drift is caught, known noise is ignored
# ──────────────────────────────────────────────────────────────────────────────


def _bernoulli_suite(rng: random.Random, p: float, n_cases: int = 200) -> list[float]:
    """One simulated suite run: n_cases pass/fail scores with pass-rate p."""
    return [1.0 if rng.random() < p else 0.0 for _ in range(n_cases)]


def test_synthetic_real_drift_is_caught_and_noise_is_ignored() -> None:
    rng = random.Random(42)  # deterministic; assertions verified for this stream
    true_p = 0.9

    baseline = summarize_run({"exact": [_bernoulli_suite(rng, true_p) for _ in range(8)]})
    stats = baseline.scorers["exact"]
    # Sanity: the noise floor brackets the true pass-rate.
    assert stats.ci_low < true_p < stats.ci_high

    # Candidate drawn from the SAME distribution → fluctuation, not drift.
    noise_candidate = sum(_bernoulli_suite(rng, true_p)) / 200
    assert not compare_to_baseline(baseline, {"exact": noise_candidate}).regressed

    # Candidate from a genuinely degraded model (p = 0.75) → real drift.
    drift_candidate = sum(_bernoulli_suite(rng, 0.75)) / 200
    result = compare_to_baseline(baseline, {"exact": drift_candidate})
    assert result.regressed
    assert result.scorers[0].delta < -0.05
