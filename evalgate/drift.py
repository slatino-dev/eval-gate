"""Drift detection — compare current eval results against a baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


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
    """Comparison between a baseline and current run."""

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
    """
    Return a DriftReport. ``regressed`` is True when pass_rate dropped
    by more than ``threshold`` (e.g. 0.05 = 5 pp).
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
