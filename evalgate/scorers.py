"""Scorers — each maps ``(actual, expected)`` to a score in [0, 1] + details.

All scorers are deterministic and embedding-free except :class:`JudgeScorer`,
which calls an OpenAI-compatible endpoint and degrades to a visible
"skipped (no endpoint)" result when none is configured.
"""

from __future__ import annotations

import json
import math
import os
import re
import string
from collections import Counter
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from .adapters import ChatMessage, CompletionRequest, OpenAIAdapter

JsonValue = None | bool | int | float | str | list[Any] | dict[str, Any]


class ScoreResult(BaseModel):
    """Outcome of scoring one case."""

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    skipped: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Scorer(Protocol):
    """Anything with a ``score(actual, expected) -> ScoreResult`` method."""

    def score(self, actual: str, expected: Any) -> ScoreResult: ...


def _no_expected() -> ScoreResult:
    return ScoreResult(
        score=0.0, passed=False, details={"reason": "case has no 'expected' value"}
    )


def _truncate(value: Any, limit: int = 200) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic string scorers
# ──────────────────────────────────────────────────────────────────────────────


class ExactMatchScorer:
    """1.0 iff actual equals expected after whitespace strip; else 0.0."""

    def score(self, actual: str, expected: Any) -> ScoreResult:
        if expected is None:
            return _no_expected()
        matched = actual.strip() == str(expected).strip()
        details: dict[str, Any] = {}
        if not matched:
            details = {"actual": _truncate(actual.strip()), "expected": _truncate(expected)}
        return ScoreResult(score=1.0 if matched else 0.0, passed=matched, details=details)


class ContainsScorer:
    """1.0 iff expected substring occurs in actual (case-insensitive)."""

    def score(self, actual: str, expected: Any) -> ScoreResult:
        if expected is None:
            return _no_expected()
        found = str(expected).lower() in actual.lower()
        return ScoreResult(score=1.0 if found else 0.0, passed=found)


class RegexScorer:
    """1.0 iff the pattern matches anywhere in actual (``re.search``).

    The pattern comes from the constructor, or — for golden datasets — from
    the case's ``expected`` field when no constructor pattern was given.
    """

    def __init__(self, pattern: str | None = None, flags: int = 0) -> None:
        self._pattern = re.compile(pattern, flags) if pattern is not None else None
        self._flags = flags

    def score(self, actual: str, expected: Any) -> ScoreResult:
        pattern = self._pattern
        if pattern is None:
            if not isinstance(expected, str) or not expected:
                return ScoreResult(
                    score=0.0,
                    passed=False,
                    details={"reason": "no regex pattern (pass one to the constructor or via 'expected')"},
                )
            try:
                pattern = re.compile(expected, self._flags)
            except re.error as exc:
                return ScoreResult(
                    score=0.0, passed=False, details={"reason": f"invalid regex: {exc}"}
                )
        match = pattern.search(actual)
        details = {"pattern": pattern.pattern}
        if match is not None:
            details["match"] = _truncate(match.group(0))
        return ScoreResult(score=1.0 if match else 0.0, passed=match is not None, details=details)


# ──────────────────────────────────────────────────────────────────────────────
# json_subset — deep subset match (expected ⊆ actual) with partial credit
# ──────────────────────────────────────────────────────────────────────────────


class JsonSubsetScorer:
    """Deep subset match: every assertion in ``expected`` must hold in actual.

    Semantics:

    - **objects** — every expected key must exist in actual and match
      recursively; extra keys in actual are allowed;
    - **arrays** — compared element-wise by index; expected may be a prefix
      of actual, never longer;
    - **leaves** — values must be equal. Numbers compare numerically
      (``1 == 1.0``) but bool never equals int/float, and ``"1"`` never
      equals ``1``. ``null`` requires the key to be present with ``null``;
    - **empty containers** — ``{}`` / ``[]`` assert "a container of this
      type exists here" and count as one assertion.

    Score is the fraction of expected leaf assertions that hold (partial
    credit); ``passed`` requires all of them. ``details["mismatches"]``
    lists the failing paths with reasons.
    """

    def score(self, actual: str, expected: Any) -> ScoreResult:
        if expected is None:
            return _no_expected()
        parsed = self._parse_actual(actual)
        if isinstance(parsed, _ParseFailure):
            return ScoreResult(
                score=0.0, passed=False, details={"reason": parsed.reason}
            )
        mismatches: list[dict[str, Any]] = []
        matched, total = _subset_match(expected, parsed.value, "$", mismatches)
        score = matched / total if total else 1.0
        return ScoreResult(
            score=score,
            passed=matched == total,
            details={"matched": matched, "total": total, "mismatches": mismatches},
        )

    @staticmethod
    def _parse_actual(actual: Any) -> _ParsedActual | _ParseFailure:
        if not isinstance(actual, str):
            return _ParsedActual(value=actual)
        text = actual.strip()
        # Tolerate a markdown code fence around the JSON body.
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
        if fenced is not None:
            text = fenced.group(1)
        try:
            return _ParsedActual(value=json.loads(text))
        except json.JSONDecodeError as exc:
            return _ParseFailure(reason=f"actual is not valid JSON: {exc.msg}")


class _ParsedActual(BaseModel):
    value: Any


class _ParseFailure(BaseModel):
    reason: str


def _subset_match(
    expected: Any,
    actual: Any,
    path: str,
    mismatches: list[dict[str, Any]],
) -> tuple[int, int]:
    """Recursively match expected ⊆ actual; return (matched, total) assertions."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append(
                {"path": path, "reason": f"expected object, got {type(actual).__name__}"}
            )
            return 0, _assertion_count(expected)
        if not expected:
            return 1, 1
        matched = total = 0
        for key, exp_val in expected.items():
            child = f"{path}.{key}"
            if key not in actual:
                mismatches.append({"path": child, "reason": "missing key"})
                total += _assertion_count(exp_val)
            else:
                m, t = _subset_match(exp_val, actual[key], child, mismatches)
                matched += m
                total += t
        return matched, total

    if isinstance(expected, list):
        if not isinstance(actual, list):
            mismatches.append(
                {"path": path, "reason": f"expected array, got {type(actual).__name__}"}
            )
            return 0, _assertion_count(expected)
        if not expected:
            return 1, 1
        matched = total = 0
        for index, exp_val in enumerate(expected):
            child = f"{path}[{index}]"
            if index >= len(actual):
                mismatches.append(
                    {"path": child, "reason": f"missing index (actual has {len(actual)} items)"}
                )
                total += _assertion_count(exp_val)
            else:
                m, t = _subset_match(exp_val, actual[index], child, mismatches)
                matched += m
                total += t
        return matched, total

    if _leaf_equal(expected, actual):
        return 1, 1
    mismatches.append(
        {
            "path": path,
            "reason": "value mismatch",
            "expected": _truncate(expected),
            "actual": _truncate(actual),
        }
    )
    return 0, 1


def _assertion_count(value: Any) -> int:
    """Number of leaf assertions in an expected subtree (empty containers = 1)."""
    if isinstance(value, dict):
        return sum(_assertion_count(v) for v in value.values()) if value else 1
    if isinstance(value, list):
        return sum(_assertion_count(v) for v in value) if value else 1
    return 1


def _leaf_equal(expected: Any, actual: Any) -> bool:
    # bool is an int subclass in Python — keep JSON true/false distinct from 1/0.
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return float(expected) == float(actual)
    if type(expected) is not type(actual):
        return False
    return bool(expected == actual)


# ──────────────────────────────────────────────────────────────────────────────
# numeric_tolerance
# ──────────────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


class NumericToleranceScorer:
    """Match a numeric answer within absolute and/or relative tolerance.

    Pass iff ``|actual - expected| <= max(rel_tol * |expected|, abs_tol)``.
    The relative tolerance is anchored on the golden value (the reference),
    not on ``max(|a|, |b|)`` as in :func:`math.isclose`. With both tolerances
    at their default 0, this is exact numeric equality.

    If ``actual`` is a string that is not itself a number, the first numeric
    token is extracted (noted in details as ``extracted_from_text``).
    """

    def __init__(self, abs_tol: float = 0.0, rel_tol: float = 0.0) -> None:
        if abs_tol < 0 or rel_tol < 0:
            raise ValueError("tolerances must be non-negative")
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def score(self, actual: str, expected: Any) -> ScoreResult:
        if expected is None:
            return _no_expected()
        expected_value = _coerce_number(expected)
        if expected_value is None:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"reason": f"expected value is not numeric: {_truncate(expected)}"},
            )
        actual_value, extracted = self._extract_number(actual)
        if actual_value is None:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"reason": f"no number found in actual: {_truncate(actual)}"},
            )
        abs_error = abs(actual_value - expected_value)
        tolerance = max(self.rel_tol * abs(expected_value), self.abs_tol)
        ok = abs_error <= tolerance
        details: dict[str, Any] = {
            "expected": expected_value,
            "actual": actual_value,
            "abs_error": abs_error,
            "tolerance": tolerance,
        }
        if expected_value != 0:
            details["rel_error"] = abs_error / abs(expected_value)
        if extracted:
            details["extracted_from_text"] = True
        return ScoreResult(score=1.0 if ok else 0.0, passed=ok, details=details)

    @staticmethod
    def _extract_number(actual: Any) -> tuple[float | None, bool]:
        value = _coerce_number(actual)
        if value is not None:
            return value, False
        if isinstance(actual, str):
            match = _NUMBER_RE.search(actual)
            if match is not None:
                return float(match.group(0)), True
        return None, False


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


# ──────────────────────────────────────────────────────────────────────────────
# text_similarity — token-level F1, embedding-free
# ──────────────────────────────────────────────────────────────────────────────

_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})


def _tokenize(text: str) -> list[str]:
    """Lowercase, replace punctuation with spaces, split on whitespace."""
    return text.lower().translate(_PUNCT_TABLE).split()


class TextSimilarityScorer:
    """Token-level F1 between actual and expected (SQuAD-style, no embeddings).

    Both texts are lowercased, punctuation-stripped, and whitespace-tokenized
    (articles are NOT removed — the metric stays trivially hand-checkable).
    Overlap is the multiset intersection of token counts:
    ``P = overlap/|actual|``, ``R = overlap/|expected|``, ``F1 = 2PR/(P+R)``.

    ``passed`` is ``F1 >= pass_threshold`` (default 0.5).
    """

    def __init__(self, pass_threshold: float = 0.5) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be in [0, 1]")
        self.pass_threshold = pass_threshold

    def score(self, actual: str, expected: Any) -> ScoreResult:
        if expected is None:
            return _no_expected()
        pred_tokens = _tokenize(actual)
        gold_tokens = _tokenize(str(expected))
        if not pred_tokens and not gold_tokens:
            return ScoreResult(score=1.0, passed=True, details={"reason": "both texts empty"})
        if not pred_tokens or not gold_tokens:
            return ScoreResult(score=0.0, passed=False, details={"reason": "one text is empty"})
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if overlap == 0:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"precision": 0.0, "recall": 0.0, "overlap": 0},
            )
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return ScoreResult(
            score=f1,
            passed=f1 >= self.pass_threshold,
            details={
                "precision": precision,
                "recall": recall,
                "overlap": overlap,
                "pred_tokens": len(pred_tokens),
                "gold_tokens": len(gold_tokens),
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# judge — LLM-as-judge via any OpenAI-compatible endpoint (optional)
# ──────────────────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluation judge. Compare the candidate answer to the "
    "reference answer and rate semantic correctness. Respond with ONLY a JSON "
    'object: {"score": <float between 0 and 1>, "reason": "<one sentence>"}.'
)

_JUDGE_USER_TEMPLATE = (
    "Reference answer:\n{expected}\n\nCandidate answer:\n{actual}\n\n"
    "Rate the candidate against the reference. JSON only."
)


class JudgeScorer:
    """LLM-as-judge. Optional: degrades to a visible skip without an endpoint.

    The endpoint is resolved from ``base_url``, then the ``OPENAI_BASE_URL``
    or ``LLM_BASE_URL`` environment variables. If none is set, ``score``
    returns ``skipped=True`` with ``details["status"] == "skipped (no
    endpoint)"`` instead of raising — judge cases never silently count as
    failures, and never block offline runs. Transport/parse errors likewise
    come back as visible error results, not exceptions.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        pass_threshold: float = 0.5,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get(
            "LLM_BASE_URL"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self.model = model or os.environ.get("OPENAI_MODEL") or "default"
        self.pass_threshold = pass_threshold
        self.timeout = timeout

    def score(self, actual: str, expected: Any) -> ScoreResult:
        if self.base_url is None:
            return ScoreResult(
                score=0.0,
                passed=False,
                skipped=True,
                details={
                    "status": "skipped (no endpoint)",
                    "reason": "set OPENAI_BASE_URL or LLM_BASE_URL, or pass base_url=",
                },
            )
        if expected is None:
            reference = "(no reference provided — judge on standalone quality)"
        else:
            reference = str(expected)
        request = CompletionRequest(
            model=self.model,
            messages=[
                ChatMessage(role="system", content=_JUDGE_SYSTEM_PROMPT),
                ChatMessage(
                    role="user",
                    content=_JUDGE_USER_TEMPLATE.format(expected=reference, actual=actual),
                ),
            ],
        )
        adapter = OpenAIAdapter(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        try:
            response = adapter.complete(request)
        except Exception as exc:  # noqa: BLE001 — judge failures must be visible, not fatal
            return ScoreResult(
                score=0.0,
                passed=False,
                details={"status": "error", "reason": f"judge call failed: {exc}"},
            )
        verdict = _parse_judge_verdict(response.content)
        if verdict is None:
            return ScoreResult(
                score=0.0,
                passed=False,
                details={
                    "status": "error",
                    "reason": "could not parse a score from the judge response",
                    "raw": _truncate(response.content),
                },
            )
        score, reason = verdict
        return ScoreResult(
            score=score,
            passed=score >= self.pass_threshold,
            details={"status": "judged", "judge_model": self.model, "reason": reason},
        )


def _parse_judge_verdict(content: str) -> tuple[float, str] | None:
    """Extract (score, reason) from a judge reply; clamp score into [0, 1]."""
    json_match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if json_match is not None:
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            raw_score = payload.get("score")
            if isinstance(raw_score, int | float) and not isinstance(raw_score, bool):
                reason = payload.get("reason")
                return _clamp01(float(raw_score)), str(reason) if reason else ""
    # Fallback: the first bare number in the reply.
    number_match = _NUMBER_RE.search(content)
    if number_match is not None:
        return _clamp01(float(number_match.group(0))), "parsed bare number from reply"
    return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

SCORER_REGISTRY: dict[str, Callable[..., Scorer]] = {
    "exact": ExactMatchScorer,
    "contains": ContainsScorer,
    "regex": RegexScorer,
    "json_subset": JsonSubsetScorer,
    "numeric_tolerance": NumericToleranceScorer,
    "text_similarity": TextSimilarityScorer,
    "judge": JudgeScorer,
}


def get_scorer(name: str, **kwargs: Any) -> Scorer:
    """Lookup a scorer by registry name."""
    if name not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scorer '{name}'. Available: {sorted(SCORER_REGISTRY)}")
    return SCORER_REGISTRY[name](**kwargs)
