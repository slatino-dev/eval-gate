"""Scoring functions for evaluating LLM outputs."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ScoreResult(BaseModel):
    passed: bool
    score: float
    details: dict[str, Any] = {}


class Scorer(ABC):
    """Base class for all scorers."""

    @abstractmethod
    def score(self, output: str, expected: str | None) -> ScoreResult: ...


class ExactMatchScorer(Scorer):
    """Pass iff output matches expected exactly (after strip)."""

    def score(self, output: str, expected: str | None) -> ScoreResult:
        if expected is None:
            return ScoreResult(passed=False, score=0.0, details={"reason": "no expected"})
        matched = output.strip() == expected.strip()
        return ScoreResult(passed=matched, score=1.0 if matched else 0.0)


class ContainsScorer(Scorer):
    """Pass iff expected substring is found in output (case-insensitive)."""

    def score(self, output: str, expected: str | None) -> ScoreResult:
        if expected is None:
            return ScoreResult(passed=False, score=0.0, details={"reason": "no expected"})
        found = expected.lower() in output.lower()
        return ScoreResult(passed=found, score=1.0 if found else 0.0)


class RegexScorer(Scorer):
    """Pass iff output matches the provided regex pattern."""

    def __init__(self, pattern: str, flags: int = re.IGNORECASE) -> None:
        self.pattern = re.compile(pattern, flags)

    def score(self, output: str, expected: str | None) -> ScoreResult:
        matched = bool(self.pattern.search(output))
        return ScoreResult(
            passed=matched,
            score=1.0 if matched else 0.0,
            details={"pattern": self.pattern.pattern},
        )


SCORER_REGISTRY: dict[str, type[Scorer]] = {
    "exact": ExactMatchScorer,
    "contains": ContainsScorer,
}


def get_scorer(name: str, **kwargs: Any) -> Scorer:
    """Lookup a scorer by registry name."""
    if name not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scorer '{name}'. Available: {list(SCORER_REGISTRY)}")
    return SCORER_REGISTRY[name](**kwargs)
