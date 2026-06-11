"""Hand-checked unit tests for the scorer math.

Every expected score in this file was computed by hand first; the assertions
encode those derivations (shown in comments where non-obvious).
"""

from __future__ import annotations

import httpx
import pytest
from pytest_httpx import HTTPXMock

from evalgate.scorers import (
    ExactMatchScorer,
    JsonSubsetScorer,
    JudgeScorer,
    NumericToleranceScorer,
    RegexScorer,
    Scorer,
    TextSimilarityScorer,
    get_scorer,
)

# ──────────────────────────────────────────────────────────────────────────────
# exact / regex
# ──────────────────────────────────────────────────────────────────────────────


def test_exact_strips_whitespace() -> None:
    result = ExactMatchScorer().score("  Paris\n", "Paris")
    assert result.passed and result.score == 1.0


def test_exact_mismatch_includes_both_sides_in_details() -> None:
    result = ExactMatchScorer().score("Lyon", "Paris")
    assert not result.passed and result.score == 0.0
    assert result.details == {"actual": "Lyon", "expected": "Paris"}


def test_exact_no_expected_is_zero_with_reason() -> None:
    result = ExactMatchScorer().score("anything", None)
    assert result.score == 0.0 and "expected" in result.details["reason"]


def test_regex_pattern_from_expected_field() -> None:
    result = RegexScorer().score("2026-06-11", r"^\d{4}-\d{2}-\d{2}$")
    assert result.passed and result.details["match"] == "2026-06-11"


def test_regex_constructor_pattern_wins() -> None:
    scorer = RegexScorer(pattern=r"\bhello\b")
    assert scorer.score("well hello there", "ignored-expected").passed
    assert not scorer.score("goodbye", "hello").passed


def test_regex_invalid_pattern_is_visible_failure() -> None:
    result = RegexScorer().score("text", "([unclosed")
    assert result.score == 0.0 and "invalid regex" in result.details["reason"]


# ──────────────────────────────────────────────────────────────────────────────
# json_subset — deep subset semantics, hand-checked
# ──────────────────────────────────────────────────────────────────────────────


def test_json_subset_ignores_extra_keys() -> None:
    # expected has 2 leaf assertions (a, b.c); both hold → 2/2 = 1.0
    result = JsonSubsetScorer().score(
        '{"a": 1, "b": {"c": 2}, "extra": 9}', {"a": 1, "b": {"c": 2}}
    )
    assert result.passed and result.score == 1.0
    assert result.details == {"matched": 2, "total": 2, "mismatches": []}


def test_json_subset_missing_nested_key_gives_partial_credit() -> None:
    # leaves: a (match), b.c (missing) → 1/2 = 0.5
    result = JsonSubsetScorer().score('{"a": 1, "b": {}}', {"a": 1, "b": {"c": 2}})
    assert not result.passed
    assert result.score == 0.5
    assert result.details["mismatches"] == [{"path": "$.b.c", "reason": "missing key"}]


def test_json_subset_partial_credit_fraction() -> None:
    # a match, b match, c value mismatch, d missing → 2/4 = 0.5
    result = JsonSubsetScorer().score(
        '{"a": 1, "b": 2, "c": 999}', {"a": 1, "b": 2, "c": 3, "d": 4}
    )
    assert result.score == 0.5
    assert result.details["matched"] == 2 and result.details["total"] == 4


def test_json_subset_bool_never_equals_int() -> None:
    assert not JsonSubsetScorer().score('{"flag": 1}', {"flag": True}).passed
    assert not JsonSubsetScorer().score('{"n": true}', {"n": 1}).passed


def test_json_subset_int_and_float_compare_numerically() -> None:
    assert JsonSubsetScorer().score('{"x": 1.0}', {"x": 1}).passed


def test_json_subset_string_number_never_equals_number() -> None:
    assert not JsonSubsetScorer().score('{"x": "1"}', {"x": 1}).passed


def test_json_subset_null_requires_present_null() -> None:
    assert JsonSubsetScorer().score('{"a": null}', {"a": None}).passed
    result = JsonSubsetScorer().score("{}", {"a": None})
    assert not result.passed
    assert result.details["mismatches"][0]["reason"] == "missing key"


def test_json_subset_list_prefix_allowed_but_not_longer() -> None:
    assert JsonSubsetScorer().score('{"xs": [1, 2, 3]}', {"xs": [1, 2]}).passed
    # expected longer than actual: indices 0,1 match, index 2 missing → 2/3
    result = JsonSubsetScorer().score("[1, 2]", [1, 2, 3])
    assert result.score == pytest.approx(2 / 3)
    assert result.details["mismatches"][0]["path"] == "$[2]"


def test_json_subset_list_order_matters() -> None:
    result = JsonSubsetScorer().score("[1, 2]", [2, 1])
    assert result.score == 0.0  # element-wise by index: both positions mismatch


def test_json_subset_empty_containers_assert_type() -> None:
    assert JsonSubsetScorer().score('{"x": 1}', {}).passed          # {} ⊆ any object
    assert JsonSubsetScorer().score('{"a": []}', {"a": []}).passed
    result = JsonSubsetScorer().score('{"a": []}', {"a": {}})        # array is not object
    assert result.score == 0.0
    assert "expected object" in result.details["mismatches"][0]["reason"]


def test_json_subset_invalid_actual_json_is_visible() -> None:
    result = JsonSubsetScorer().score("not json at all", {"a": 1})
    assert result.score == 0.0
    assert "not valid JSON" in result.details["reason"]


def test_json_subset_tolerates_markdown_fence() -> None:
    assert JsonSubsetScorer().score('```json\n{"a": 1}\n```', {"a": 1}).passed


def test_json_subset_accepts_pre_parsed_actual() -> None:
    assert JsonSubsetScorer().score({"a": 1, "b": 2}, {"a": 1}).passed  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# numeric_tolerance — hand-checked
# ──────────────────────────────────────────────────────────────────────────────


def test_numeric_default_is_exact_equality() -> None:
    scorer = NumericToleranceScorer()
    assert scorer.score("4", 4).passed
    assert scorer.score("4.0", 4).passed
    assert not scorer.score("5", 4).passed


def test_numeric_abs_tolerance() -> None:
    # |101 - 100| = 1 ≤ abs_tol 2 → pass; 1 > abs_tol 0.5 → fail
    assert NumericToleranceScorer(abs_tol=2).score("101", 100).passed
    assert not NumericToleranceScorer(abs_tol=0.5).score("101", 100).passed


def test_numeric_rel_tolerance_anchored_on_expected() -> None:
    # tol = 0.02 * |100| = 2; |101.9 - 100| = 1.9 ≤ 2 → pass; 3 > 2 → fail
    assert NumericToleranceScorer(rel_tol=0.02).score("101.9", 100).passed
    assert not NumericToleranceScorer(rel_tol=0.02).score("103", 100).passed


def test_numeric_combined_takes_the_looser_tolerance() -> None:
    # max(rel 0.01*100=1, abs 3) = 3; error 2.5 ≤ 3 → pass
    assert NumericToleranceScorer(abs_tol=3, rel_tol=0.01).score("102.5", 100).passed


def test_numeric_expected_zero_uses_abs_only() -> None:
    # rel_tol * |0| = 0, so only abs_tol can admit error
    assert not NumericToleranceScorer(rel_tol=0.5).score("0.001", 0).passed
    assert NumericToleranceScorer(abs_tol=0.01).score("0.001", 0).passed


def test_numeric_extracts_first_number_from_prose() -> None:
    result = NumericToleranceScorer().score("The answer is 42.", 42)
    assert result.passed
    assert result.details["extracted_from_text"] is True


def test_numeric_no_number_in_actual_is_visible() -> None:
    result = NumericToleranceScorer().score("no digits here", 42)
    assert result.score == 0.0 and "no number found" in result.details["reason"]


def test_numeric_non_numeric_expected_is_visible() -> None:
    result = NumericToleranceScorer().score("42", "not-a-number")
    assert result.score == 0.0 and "not numeric" in result.details["reason"]


def test_numeric_details_report_errors() -> None:
    result = NumericToleranceScorer(abs_tol=2).score("101", 100)
    assert result.details["abs_error"] == pytest.approx(1.0)
    assert result.details["rel_error"] == pytest.approx(0.01)


def test_numeric_negative_tolerance_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NumericToleranceScorer(abs_tol=-1)


# ──────────────────────────────────────────────────────────────────────────────
# text_similarity — token-level F1 on known pairs
# ──────────────────────────────────────────────────────────────────────────────


def test_f1_identical_after_normalization() -> None:
    result = TextSimilarityScorer().score("The cat sat on the mat.", "the cat sat on the mat")
    assert result.score == pytest.approx(1.0) and result.passed


def test_f1_known_pair_two_thirds() -> None:
    # tokens: pred {the, cat, sat}, gold {the, cat, ran}; overlap 2
    # P = 2/3, R = 2/3, F1 = 2·(2/3)(2/3)/(4/3) = 2/3
    result = TextSimilarityScorer().score("the cat sat", "the cat ran")
    assert result.score == pytest.approx(2 / 3)
    assert result.details["precision"] == pytest.approx(2 / 3)
    assert result.details["recall"] == pytest.approx(2 / 3)


def test_f1_short_answer_vs_longer_gold() -> None:
    # pred [paris], gold [paris, france]; overlap 1 → P = 1, R = 1/2, F1 = 2/3
    result = TextSimilarityScorer().score("Paris", "Paris, France")
    assert result.score == pytest.approx(2 / 3)
    assert result.details["precision"] == pytest.approx(1.0)
    assert result.details["recall"] == pytest.approx(0.5)


def test_f1_multiset_overlap_counts_duplicates_correctly() -> None:
    # pred [a, a, b], gold [a, b, b]; overlap = min(2,1) + min(1,2) = 2
    # P = 2/3, R = 2/3 → F1 = 2/3
    result = TextSimilarityScorer().score("a a b", "a b b")
    assert result.score == pytest.approx(2 / 3)
    assert result.details["overlap"] == 2


def test_f1_disjoint_texts_zero() -> None:
    result = TextSimilarityScorer().score("alpha beta", "gamma delta")
    assert result.score == 0.0 and not result.passed


def test_f1_both_empty_after_normalization_is_perfect() -> None:
    assert TextSimilarityScorer().score("!!!", "...").score == 1.0


def test_f1_one_empty_is_zero() -> None:
    assert TextSimilarityScorer().score("", "the answer").score == 0.0


def test_f1_threshold_gates_passed_not_score() -> None:
    result = TextSimilarityScorer(pass_threshold=0.7).score("the cat sat", "the cat ran")
    assert result.score == pytest.approx(2 / 3)
    assert not result.passed


# ──────────────────────────────────────────────────────────────────────────────
# judge — optional, degrades visibly without an endpoint
# ──────────────────────────────────────────────────────────────────────────────


def test_judge_without_endpoint_is_visibly_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    result = JudgeScorer().score("candidate", "reference")
    assert result.skipped is True
    assert result.score == 0.0 and not result.passed
    assert result.details["status"] == "skipped (no endpoint)"


def test_judge_resolves_endpoint_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://judge.test")
    assert JudgeScorer().base_url == "http://judge.test"


def _judge_response(content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "model": "judge-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {},
    }


def test_judge_parses_json_verdict(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://judge.test/v1/chat/completions",
        json=_judge_response('{"score": 0.8, "reason": "mostly equivalent"}'),
    )
    result = JudgeScorer(base_url="http://judge.test", model="judge-model").score("c", "r")
    assert result.score == pytest.approx(0.8)
    assert result.passed and not result.skipped
    assert result.details["status"] == "judged"


def test_judge_clamps_out_of_range_score(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://judge.test/v1/chat/completions",
        json=_judge_response('{"score": 1.5, "reason": "overenthusiastic"}'),
    )
    result = JudgeScorer(base_url="http://judge.test").score("c", "r")
    assert result.score == 1.0


def test_judge_falls_back_to_bare_number(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://judge.test/v1/chat/completions",
        json=_judge_response("0.75"),
    )
    result = JudgeScorer(base_url="http://judge.test").score("c", "r")
    assert result.score == pytest.approx(0.75)


def test_judge_unparseable_reply_is_visible_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="http://judge.test/v1/chat/completions",
        json=_judge_response("looks good to me"),
    )
    result = JudgeScorer(base_url="http://judge.test").score("c", "r")
    assert result.score == 0.0
    assert result.details["status"] == "error"


def test_judge_transport_error_does_not_raise(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    result = JudgeScorer(base_url="http://judge.test").score("c", "r")
    assert result.score == 0.0
    assert result.details["status"] == "error"
    assert "judge call failed" in result.details["reason"]


# ──────────────────────────────────────────────────────────────────────────────
# registry / protocol
# ──────────────────────────────────────────────────────────────────────────────


def test_registry_exposes_all_scorers() -> None:
    for name in ("exact", "contains", "regex", "json_subset", "numeric_tolerance",
                 "text_similarity", "judge"):
        assert isinstance(get_scorer(name), Scorer)


def test_registry_passes_kwargs_through() -> None:
    scorer = get_scorer("numeric_tolerance", abs_tol=2)
    assert scorer.score("101", 100).passed


def test_registry_unknown_name_lists_available() -> None:
    with pytest.raises(ValueError, match="json_subset"):
        get_scorer("nonexistent")
