"""Tests for golden-dataset JSONL loading and validation errors."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalgate.dataset import DatasetError, EvalDataset, load_jsonl

VALID_JSONL = """\
{"id": "a", "input": "What is 2+2?", "expected": "4"}
{"id": "b", "input": "Return JSON.", "expected": {"x": 1}, "meta": {"scorer": "json_subset"}}

{"id": "c", "input": "Pi?", "expected": 3.14, "tags": ["math"]}
"""


def _write(tmp_path: Path, text: str, name: str = "golden.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_jsonl_happy_path(tmp_path: Path) -> None:
    cases = load_jsonl(_write(tmp_path, VALID_JSONL))
    assert [c.id for c in cases] == ["a", "b", "c"]
    assert cases[0].expected == "4"
    assert cases[1].expected == {"x": 1}          # objects survive as objects
    assert cases[1].meta == {"scorer": "json_subset"}
    assert cases[2].expected == 3.14              # numbers survive as numbers
    assert cases[0].meta == {}                    # meta defaults to empty
    assert cases[2].tags == ["math"]


def test_load_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    cases = load_jsonl(_write(tmp_path, '\n{"id": "only", "input": "hi"}\n\n'))
    assert len(cases) == 1


def test_invalid_json_reports_line_number(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"id": "a", "input": "ok"}\n{not json}\n')
    with pytest.raises(DatasetError, match=r":2: invalid JSON"):
        load_jsonl(path)


def test_non_object_line_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, '["not", "an", "object"]\n')
    with pytest.raises(DatasetError, match=r":1: each line must be a JSON object, got list"):
        load_jsonl(path)


def test_missing_required_field_names_the_field(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"id": "a"}\n')
    with pytest.raises(DatasetError, match=r":1: .*field 'input'"):
        load_jsonl(path)


def test_unknown_field_typo_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"id": "a", "input": "x", "expectd": "oops"}\n')
    with pytest.raises(DatasetError, match=r"expectd"):
        load_jsonl(path)


def test_empty_id_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"id": "", "input": "x"}\n')
    with pytest.raises(DatasetError, match=r"field 'id'"):
        load_jsonl(path)


def test_duplicate_id_reports_both_lines(tmp_path: Path) -> None:
    path = _write(tmp_path, '{"id": "dup", "input": "x"}\n{"id": "dup", "input": "y"}\n')
    with pytest.raises(DatasetError, match=r":2: duplicate case id 'dup' \(first seen on line 1\)"):
        load_jsonl(path)


def test_empty_dataset_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "\n\n")
    with pytest.raises(DatasetError, match=r"contains no cases"):
        load_jsonl(path)


def test_missing_file_clear_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match=r"not found"):
        load_jsonl(tmp_path / "nope.jsonl")


def test_from_jsonl_names_dataset_after_file_stem(tmp_path: Path) -> None:
    ds = EvalDataset.from_jsonl(_write(tmp_path, VALID_JSONL, name="regression_suite.jsonl"))
    assert ds.name == "regression_suite"
    assert len(ds.cases) == 3


def test_repo_example_golden_dataset_loads() -> None:
    example = Path(__file__).parent.parent / "examples" / "golden_dataset.jsonl"
    cases = load_jsonl(example)
    assert len(cases) >= 4
    assert all(c.id for c in cases)
