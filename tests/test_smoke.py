"""Smoke tests — verify core evalgate imports and basic logic."""

from __future__ import annotations

from evalgate import __version__
from evalgate.dataset import EvalCase, EvalDataset
from evalgate.drift import RunSummary, detect_drift
from evalgate.scorers import ContainsScorer, ExactMatchScorer, get_scorer


def test_version_exists() -> None:
    assert __version__ == "0.1.0"


def test_exact_match_scorer_pass() -> None:
    scorer = ExactMatchScorer()
    result = scorer.score("hello world", "hello world")
    assert result.passed
    assert result.score == 1.0


def test_exact_match_scorer_fail() -> None:
    scorer = ExactMatchScorer()
    result = scorer.score("hello world", "different")
    assert not result.passed
    assert result.score == 0.0


def test_contains_scorer() -> None:
    scorer = ContainsScorer()
    result = scorer.score("The quick brown fox", "brown fox")
    assert result.passed


def test_get_scorer_unknown_raises() -> None:
    import pytest
    with pytest.raises(ValueError, match="Unknown scorer"):
        get_scorer("nonexistent")


def test_dataset_filter_by_tag() -> None:
    ds = EvalDataset(
        name="test",
        cases=[
            EvalCase(id="a", prompt="p1", tags=["math"]),
            EvalCase(id="b", prompt="p2", tags=["coding"]),
            EvalCase(id="c", prompt="p3", tags=["math", "coding"]),
        ],
    )
    filtered = ds.filter_by_tag("math")
    assert len(filtered.cases) == 2
    assert {c.id for c in filtered.cases} == {"a", "c"}


def test_drift_no_regression() -> None:
    base = RunSummary(run_id="base", dataset="d", model="m", total=100, passed=90, failed=10, pass_rate=0.9)
    curr = RunSummary(run_id="curr", dataset="d", model="m", total=100, passed=88, failed=12, pass_rate=0.88)
    report = detect_drift(base, curr, threshold=0.05)
    assert not report.regressed  # 0.02 drop < 0.05 threshold


def test_drift_regression_detected() -> None:
    base = RunSummary(run_id="base", dataset="d", model="m", total=100, passed=90, failed=10, pass_rate=0.9)
    curr = RunSummary(run_id="curr", dataset="d", model="m", total=100, passed=80, failed=20, pass_rate=0.8)
    report = detect_drift(base, curr, threshold=0.05)
    assert report.regressed  # 0.10 drop > 0.05 threshold
