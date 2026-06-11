"""Dataset loading and management for eval suites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single evaluation case."""

    id: str
    prompt: str
    expected: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A named collection of evaluation cases."""

    name: str
    version: str = "0.1.0"
    cases: list[EvalCase] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path | str) -> EvalDataset:
        """Load a dataset from a YAML file."""
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls.model_validate(raw)

    def filter_by_tag(self, tag: str) -> EvalDataset:
        """Return a new dataset containing only cases with the given tag."""
        filtered = [c for c in self.cases if tag in c.tags]
        return self.model_copy(update={"cases": filtered})
