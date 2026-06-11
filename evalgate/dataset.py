"""Golden datasets — pydantic models + JSONL/YAML loaders with precise errors.

The canonical golden-dataset format is JSONL: one case per line, each an
object with the shape ``{id, input, expected, meta}``. Every loader error
includes the file path and (where it applies) the 1-based line number, so a
failing CI log points straight at the offending record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class DatasetError(ValueError):
    """A dataset file is missing, malformed, or fails schema validation."""


class EvalCase(BaseModel):
    """One golden record: ``{id, input, expected, meta}``.

    - ``id``: unique, non-empty case identifier.
    - ``input``: the prompt / payload sent to the system under test.
    - ``expected``: the golden value scorers compare against. Its type depends
      on the scorer: ``str`` for exact / text_similarity, a regex pattern
      string for regex, an object for json_subset, a number for
      numeric_tolerance. ``None`` means "no reference" (judge-only cases).
    - ``meta``: free-form metadata (scorer name, difficulty, source, ...).
    - ``tags``: optional labels used by :meth:`EvalDataset.filter_by_tag`.

    Unknown keys are rejected (``extra="forbid"``) so typos like ``expectd``
    fail loudly at load time instead of silently scoring 0.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    input: str | int | float | bool | list[Any] | dict[str, Any]
    expected: Any = None
    meta: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class EvalDataset(BaseModel):
    """A named collection of evaluation cases."""

    name: str
    version: str = "0.1.0"
    cases: list[EvalCase] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path | str) -> EvalDataset:
        """Load a suite-style dataset (name/version/cases) from a YAML file."""
        p = Path(path)
        if not p.is_file():
            raise DatasetError(f"{p}: dataset file not found")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise DatasetError(f"{p}: invalid YAML — {exc}") from exc
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise DatasetError(f"{p}: {_format_validation_error(exc)}") from exc

    @classmethod
    def from_jsonl(cls, path: Path | str, name: str | None = None) -> EvalDataset:
        """Load a golden JSONL dataset; the name defaults to the file stem."""
        p = Path(path)
        return cls(name=name or p.stem, cases=load_jsonl(p))

    def filter_by_tag(self, tag: str) -> EvalDataset:
        """Return a new dataset containing only cases with the given tag."""
        filtered = [c for c in self.cases if tag in c.tags]
        return self.model_copy(update={"cases": filtered})


def load_jsonl(path: Path | str) -> list[EvalCase]:
    """Load golden cases from a JSONL file (one JSON object per line).

    Blank lines are skipped. Raises :class:`DatasetError` with the file path
    and line number on: unreadable file, invalid JSON, a line that is not an
    object, schema violations, duplicate ids, or an empty dataset.
    """
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"{p}: dataset file not found")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"{p}: cannot read file — {exc}") from exc

    cases: list[EvalCase] = []
    seen: dict[str, int] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{p}:{lineno}: invalid JSON — {exc.msg} (column {exc.colno})") from exc
        if not isinstance(raw, dict):
            raise DatasetError(
                f"{p}:{lineno}: each line must be a JSON object, got {type(raw).__name__}"
            )
        try:
            case = EvalCase.model_validate(raw)
        except ValidationError as exc:
            raise DatasetError(f"{p}:{lineno}: {_format_validation_error(exc)}") from exc
        if case.id in seen:
            raise DatasetError(
                f"{p}:{lineno}: duplicate case id '{case.id}' (first seen on line {seen[case.id]})"
            )
        seen[case.id] = lineno
        cases.append(case)

    if not cases:
        raise DatasetError(f"{p}: dataset contains no cases")
    return cases


def _format_validation_error(exc: ValidationError) -> str:
    """Flatten a pydantic ValidationError into one human-readable line."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(piece) for piece in err["loc"]) or "<root>"
        parts.append(f"field '{loc}': {err['msg']}")
    return "; ".join(parts)
