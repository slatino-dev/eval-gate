"""Noise-floor drift detection — separate real regressions from sampling noise.

Workflow:

1. Run the eval suite ``k >= 3`` times against the model you trust and feed the
   per-repeat case scores to :func:`summarize_run`. That computes, per scorer,
   the mean suite score and a Student-t confidence interval over the repeat
   means — the *noise floor* — and packages it as a :class:`Baseline` you
   commit as ``baseline.json``.
2. In CI, run the candidate (ideally also k times), then call
   :func:`compare_to_baseline`. A scorer is flagged as **regressed** only when
   the candidate mean falls *below the baseline interval* — never because it
   differs from the point estimate. A candidate above the interval is flagged
   as **improved** (significant, but not a failure).

Why a t-interval and not bootstrap: k is small in CI (3–10 repeats). A
percentile bootstrap over so few atoms is badly under-covered — with k=3 there
are only 10 distinct resamples, so the interval collapses toward the data
range. Each repeat mean is itself an average over n cases, hence approximately
normal by the CLT, which is exactly the regime the small-sample t-interval is
calibrated for. The t critical value is computed exactly via the regularized
incomplete beta function (no scipy dependency) and unit-tested against
standard t-tables.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

MIN_REPEATS = 3
DEFAULT_CONFIDENCE = 0.95


class DriftError(ValueError):
    """Invalid input to the drift machinery (too few repeats, bad baseline...)."""


# ──────────────────────────────────────────────────────────────────────────────
# Student-t critical values (exact, scipy-free)
# ──────────────────────────────────────────────────────────────────────────────


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    max_iterations = 300
    eps = 3e-14
    fpmin = 1e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError("incomplete beta continued fraction failed to converge")


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_cdf(t: float, df: int) -> float:
    """CDF of the Student-t distribution with ``df`` degrees of freedom."""
    if df < 1:
        raise DriftError(f"degrees of freedom must be >= 1, got {df}")
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_critical(confidence: float, df: int) -> float:
    """Two-sided Student-t critical value (e.g. 4.3027 for 95% CI, df=2).

    Solved by bisection on the exact CDF; accurate to ~1e-10.
    """
    if not 0.0 < confidence < 1.0:
        raise DriftError(f"confidence must be in (0, 1), got {confidence}")
    target = 1.0 - (1.0 - confidence) / 2.0  # upper-tail quantile probability
    lo, hi = 0.0, 64.0
    while _student_t_cdf(hi, df) < target:
        hi *= 2.0
        if hi > 1e9:
            raise ArithmeticError("t critical value search did not bracket the target")
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _student_t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Per-scorer statistics and the committed baseline
# ──────────────────────────────────────────────────────────────────────────────


class ScorerStats(BaseModel):
    """Mean + t-based confidence interval for one scorer across k repeats."""

    scorer: str
    k: int = Field(ge=MIN_REPEATS)
    mean: float = Field(ge=0.0, le=1.0)
    std: float = Field(ge=0.0, description="sample std (ddof=1) of the repeat means")
    ci_low: float = Field(ge=0.0, le=1.0)
    ci_high: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(gt=0.0, lt=1.0)
    ci_method: str = "t"


def compute_stats(
    scorer: str,
    repeat_means: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
) -> ScorerStats:
    """Mean and two-sided t-interval over per-repeat suite means.

    ``repeat_means`` must hold one suite-mean score in [0, 1] per repeat, with
    at least :data:`MIN_REPEATS` entries. The interval is
    ``mean ± t(conf, k-1) * std / sqrt(k)``, clamped to [0, 1] because scores
    are bounded. With zero observed variance the interval collapses to a
    point — any deviation then counts as significant, which is the correct
    reading of "no noise was ever observed".
    """
    k = len(repeat_means)
    if k < MIN_REPEATS:
        raise DriftError(
            f"scorer '{scorer}': need at least {MIN_REPEATS} repeats for a "
            f"confidence interval, got k={k}"
        )
    for value in repeat_means:
        if not 0.0 <= value <= 1.0:
            raise DriftError(
                f"scorer '{scorer}': repeat means must be in [0, 1], got {value}"
            )
    mean = math.fsum(repeat_means) / k
    variance = math.fsum((v - mean) ** 2 for v in repeat_means) / (k - 1)
    std = math.sqrt(variance)
    half_width = t_critical(confidence, k - 1) * std / math.sqrt(k)
    return ScorerStats(
        scorer=scorer,
        k=k,
        mean=mean,
        std=std,
        ci_low=max(0.0, mean - half_width),
        ci_high=min(1.0, mean + half_width),
        confidence=confidence,
    )


class Baseline(BaseModel):
    """Committed noise floor: per-scorer stats from k repeats of a trusted run."""

    schema_version: int = 1
    confidence: float = Field(gt=0.0, lt=1.0, default=DEFAULT_CONFIDENCE)
    scorers: dict[str, ScorerStats]
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> Baseline:
        p = Path(path)
        if not p.is_file():
            raise DriftError(f"{p}: baseline file not found")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DriftError(f"{p}: invalid JSON — {exc.msg} (line {exc.lineno})") from exc
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise DriftError(f"{p}: invalid baseline — {exc}") from exc

    def save(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def summarize_run(
    case_scores: Mapping[str, Sequence[Sequence[float]]],
    confidence: float = DEFAULT_CONFIDENCE,
    meta: dict[str, Any] | None = None,
) -> Baseline:
    """Aggregate raw per-repeat case scores into a :class:`Baseline`.

    ``case_scores`` maps scorer name to a (k repeats) x (n cases) matrix of
    scores in [0, 1]. Skipped results (e.g. judge with no endpoint) must be
    excluded by the caller — they are not zeros. Each repeat is reduced to its
    suite mean, then :func:`compute_stats` builds the t-interval per scorer.
    """
    stats: dict[str, ScorerStats] = {}
    for scorer, repeats in case_scores.items():
        repeat_means: list[float] = []
        for index, repeat in enumerate(repeats):
            if len(repeat) == 0:
                raise DriftError(f"scorer '{scorer}': repeat {index} has no case scores")
            repeat_means.append(math.fsum(repeat) / len(repeat))
        stats[scorer] = compute_stats(scorer, repeat_means, confidence)
    return Baseline(confidence=confidence, scorers=stats, meta=meta or {})


# ──────────────────────────────────────────────────────────────────────────────
# Candidate vs. baseline comparison
# ──────────────────────────────────────────────────────────────────────────────


class ScorerDrift(BaseModel):
    """Delta of one scorer's candidate mean against the baseline interval."""

    scorer: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    regressed: bool
    improved: bool
    significant: bool


class DriftResult(BaseModel):
    """Per-scorer drift verdicts for a candidate run."""

    scorers: list[ScorerDrift]
    regressed: bool
    missing_scorers: list[str] = Field(
        default_factory=list, description="in the baseline but absent from the candidate"
    )
    new_scorers: list[str] = Field(
        default_factory=list, description="in the candidate but absent from the baseline"
    )


def compare_to_baseline(
    baseline: Baseline,
    candidate_means: Mapping[str, float],
) -> DriftResult:
    """Flag per-scorer drift of a candidate run against a committed baseline.

    A scorer **regresses** only when its candidate mean falls below the
    baseline confidence interval (``< ci_low``) — the comparison is against
    the interval, never the point estimate, so changes inside the noise floor
    are ignored. A candidate mean above ``ci_high`` is reported as
    **improved**; both directions set ``significant``.
    """
    drifts: list[ScorerDrift] = []
    missing: list[str] = []
    for name in sorted(baseline.scorers):
        stats = baseline.scorers[name]
        if name not in candidate_means:
            missing.append(name)
            continue
        candidate = candidate_means[name]
        if not 0.0 <= candidate <= 1.0:
            raise DriftError(f"scorer '{name}': candidate mean must be in [0, 1], got {candidate}")
        regressed = candidate < stats.ci_low
        improved = candidate > stats.ci_high
        drifts.append(
            ScorerDrift(
                scorer=name,
                baseline_mean=stats.mean,
                candidate_mean=candidate,
                delta=candidate - stats.mean,
                ci_low=stats.ci_low,
                ci_high=stats.ci_high,
                regressed=regressed,
                improved=improved,
                significant=regressed or improved,
            )
        )
    new = sorted(set(candidate_means) - set(baseline.scorers))
    return DriftResult(
        scorers=drifts,
        regressed=any(d.regressed for d in drifts),
        missing_scorers=missing,
        new_scorers=new,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Legacy single-run summary (used by the v0 CLI gate)
# ──────────────────────────────────────────────────────────────────────────────


class RunSummary(BaseModel):
    """Aggregated results from a single eval run."""

    run_id: str
    dataset: str
    model: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path | str) -> RunSummary:
        with open(path) as fh:
            return cls.model_validate(json.load(fh))

    def save(self, path: Path | str) -> None:
        with open(path, "w") as fh:
            json.dump(self.model_dump(), fh, indent=2)


class DriftReport(BaseModel):
    """Comparison between a baseline and current run (point-estimate gate)."""

    baseline_run_id: str
    current_run_id: str
    pass_rate_delta: float
    regressed: bool
    threshold: float
    details: dict[str, Any] = Field(default_factory=dict)


def detect_drift(
    baseline: RunSummary,
    current: RunSummary,
    threshold: float = 0.05,
) -> DriftReport:
    """Quick point-estimate gate: regressed iff pass_rate dropped > ``threshold``.

    This single-run check cannot distinguish drift from sampling noise — for
    a statistically grounded gate use :func:`summarize_run` (k >= 3 repeats)
    plus :func:`compare_to_baseline`.
    """
    delta = current.pass_rate - baseline.pass_rate
    regressed = delta < -threshold
    return DriftReport(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        pass_rate_delta=delta,
        regressed=regressed,
        threshold=threshold,
    )
