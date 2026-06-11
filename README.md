# eval-gate

Regression-eval CI gate for LLM/agent repos — catches score regressions before they merge, and tells noise apart from drift.

Status: statistical core landed (datasets, scorers, noise-floor drift detection). CLI orchestration of k-repeat runs and the GitHub Action wrapper land next.

## Why this isn't a diff-checker

LLM outputs are stochastic: the same suite scores differently run to run. A gate that compares a single run against a single baseline number flags noise as regression (flaky CI) or hides real drift inside a generous threshold. eval-gate measures the noise floor instead:

1. Run the suite `k >= 3` times against the model you trust. `evalgate.drift.summarize_run` reduces each repeat to its suite mean and computes, per scorer, a Student-t confidence interval over those means — the noise floor. Commit it as `baseline.json`.
2. In CI, score the candidate and call `compare_to_baseline`. A scorer is flagged **regressed** only when its mean falls *below the baseline interval* — never because it differs from the point estimate. Above the interval is reported as **improved**.

The t-interval (not bootstrap) is deliberate: with the 3–10 repeats realistic in CI, a percentile bootstrap is badly under-covered (k=3 has only 10 distinct resamples), while each repeat mean is already an average over n cases and hence approximately normal — exactly the regime the small-sample t-interval is calibrated for. The t critical value is computed exactly via the regularized incomplete beta function (no scipy) and unit-tested against standard t-tables.

## Golden datasets

JSONL, one case per line: `{id, input, expected, meta}`. Loader errors include the file path and line number (invalid JSON, schema violations, duplicate ids, unknown keys).

```jsonl
{"id": "extract-user-json", "input": "Return this user as JSON: Ada Lovelace, age 36, admin.", "expected": {"name": "Ada Lovelace", "age": 36, "roles": ["admin"]}, "meta": {"scorer": "json_subset"}}
{"id": "pi-two-decimals", "input": "Give pi rounded to two decimal places.", "expected": 3.14, "meta": {"scorer": "numeric_tolerance", "abs_tol": 0.005}}
```

See `examples/golden_dataset.jsonl`.

## Scorers

All return a score in `[0, 1]` plus structured details; all are deterministic and embedding-free except `judge`.

| name | semantics |
|------|-----------|
| `exact` | string equality after whitespace strip |
| `contains` | case-insensitive substring |
| `regex` | `re.search` with the pattern from config or the case's `expected` |
| `json_subset` | deep subset match `expected ⊆ actual`: extra keys allowed, arrays element-wise (prefix ok), bool ≠ int, `1 == 1.0`, partial credit = fraction of leaf assertions that hold, mismatch paths in details |
| `numeric_tolerance` | `\|a − e\| ≤ max(rel_tol·\|e\|, abs_tol)`; extracts the first number from prose if needed |
| `text_similarity` | token-level F1 (lowercase, punctuation-stripped, multiset overlap) — no embeddings |
| `judge` | LLM-as-judge against any OpenAI-compatible endpoint; **optional** — with no endpoint configured it returns a visible `skipped (no endpoint)` result, never a silent zero |

The judge endpoint resolves from `base_url=`, then `OPENAI_BASE_URL` / `LLM_BASE_URL`; the key from `OPENAI_API_KEY`.

## Quickstart (library)

```python
from evalgate.dataset import load_jsonl
from evalgate.drift import Baseline, compare_to_baseline, summarize_run
from evalgate.scorers import get_scorer

cases = load_jsonl("examples/golden_dataset.jsonl")
scorer = get_scorer("json_subset")

# k repeats of [score per case] from your trusted model:
baseline = summarize_run({"json_subset": repeat_score_matrix})
baseline.save("baseline.json")

# later, in CI:
result = compare_to_baseline(Baseline.load("baseline.json"), {"json_subset": candidate_mean})
if result.regressed:
    raise SystemExit(1)
```

## Development

```bash
pip install -e ".[dev]"
python -m ruff check . && python -m mypy evalgate/ && python -m pytest -q
python scripts/scrub_check.py   # no internal infra references in the tree
python -m mockserver.server     # local OpenAI-compatible mock for the CLI
```

## Limitations

- The CLI (`evalgate run`) currently does single-run scoring with the simple threshold gate; wiring k-repeat runs + `baseline.json` into the CLI is the next stage. The statistical core (`evalgate.drift`) is complete and tested.
- `json_subset` arrays are ordered (element-wise by index); set-like "contains any" matching is not implemented.
- `judge` quality is whatever your judge model gives you; treat it as a signal, not ground truth, and keep deterministic scorers as the primary gate.
- The noise floor assumes repeats are exchangeable (same dataset, same decoding params). Change either and the committed baseline is stale — regenerate it.
