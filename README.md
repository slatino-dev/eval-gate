# eval-gate

Block merges the moment your LLM's eval scores drift outside the noise floor — not every time a stochastic run fluctuates.

```bash
pip install evalgate

# build a k=5 noise-floor baseline from your trusted model
evalgate baseline examples/golden_dataset.jsonl \
    --base-url=http://your-endpoint/v1 --model=gpt-4o \
    --scorer=exact,json_subset --k=5 --out=baseline.json

# in CI: flag only real regressions, ignore sampling noise
evalgate run examples/golden_dataset.jsonl \
    --base-url=http://your-endpoint/v1 --model=gpt-4o \
    --scorer=exact,json_subset --k=3 --baseline=baseline.json
# exit 0 = passed  |  exit 1 = regression  |  exit 2 = usage error

# or use a local command instead of an API
evalgate run examples/command_dataset.yaml \
    --command="my-tool --query {input}" \
    --scorer=exact --k=3 --baseline=baseline.json
```

## Why point-estimates make LLM CI flaky

LLM outputs are stochastic: the same suite scores differently run to run. A gate that compares a single run against a single baseline number flags noise as regression (flaky CI) or hides real drift inside a generous threshold. eval-gate measures the noise floor instead.

**How it works:**

1. Run the suite `k >= 3` times against the model you trust. `evalgate baseline` reduces each repeat to its suite mean and computes, per scorer, a Student-t confidence interval over those means — the noise floor. Commit `baseline.json`.
2. In CI, score the candidate. A scorer is flagged **regressed** only when its mean falls *below the baseline interval* — never because it differs from the point estimate. Above the interval is reported as **improved**.

The t-interval (not bootstrap) is deliberate: with k=3–10 repeats, a percentile bootstrap is badly under-covered (k=3 has only 10 distinct resamples). Each repeat mean is already an average over n cases and hence approximately normal — exactly the regime the small-sample t-interval is calibrated for. The critical value is computed via the regularised incomplete beta function (no scipy) and unit-tested against published t-tables.

## Architecture

```
evalgate/
  cli.py        — typer CLI (run / baseline / summary subcommands)
  dataset.py    — EvalCase + EvalDataset pydantic models; JSONL + YAML loaders
  scorers.py    — Scorer protocol + 7 implementations (see table below)
  drift.py      — t-interval noise floor: summarize_run, compare_to_baseline
  report.py     — text, GitHub Step Summary markdown, and JSON report writers
  adapters.py   — OpenAIAdapter (httpx) + CommandAdapter (shell=False, no injection)

action/
  action.yml    — composite GitHub Action (installs evalgate, runs the CLI)
  src/main.ts   — Node20 entry-point: reads inputs, invokes CLI, posts sticky PR comment
  dist/main.js  — bundled JS (committed; required by node20 actions)
```

**Stack:** Python 3.11–3.13, pydantic v2, typer, httpx, PyYAML. TypeScript/Node 20 for the action layer. Zero numpy — the t-math is pure Python.

## Subcommands

### `evalgate run`

Run an eval suite against an endpoint or command, optionally comparing against a committed baseline.

```bash
evalgate run DATASET [options]

Options:
  --base-url TEXT     OpenAI-compatible API base URL  [default: http://localhost:8000]
  --model TEXT        Model name                      [default: default]
  --command TEXT      Subprocess command; use {input} for the case input
  --k INT             Repeat count: k>=3 uses statistical gate, k=1 uses point-estimate gate
  --scorer TEXT       Comma-separated scorer list     [default: contains]
  --baseline PATH     Baseline JSON from 'evalgate baseline' (k>=3) or a RunSummary (k=1)
  --threshold FLOAT   Pass-rate drop threshold for k=1 gate  [default: 0.05]
  --out PATH          Write JSON report to this path
  --github-summary    Also emit GitHub Step Summary markdown
```

### `evalgate baseline`

Run the suite k times and write a statistical noise-floor baseline.

```bash
evalgate baseline DATASET --k=5 --scorer=exact,json_subset --out=baseline.json
```

### `evalgate summary`

Pretty-print a previously saved JSON report.

```bash
evalgate summary evalgate-report.json [--github]
```

## Golden datasets

JSONL (canonical) or YAML — the CLI dispatches on file extension. One case per line in JSONL:

```jsonl
{"id": "extract-user-json", "input": "Return this user as JSON: Ada Lovelace, age 36, admin.", "expected": {"name": "Ada Lovelace", "age": 36, "roles": ["admin"]}, "meta": {"scorer": "json_subset"}}
{"id": "pi-two-decimals", "input": "Give pi rounded to two decimal places.", "expected": 3.14, "meta": {"scorer": "numeric_tolerance", "abs_tol": 0.005}}
```

Loader errors carry the file path and 1-based line number. `extra="forbid"` on the pydantic model means typos like `expectd` fail at load time, not at score time.

See `examples/golden_dataset.jsonl` and `examples/command_dataset.yaml`.

## Scorers

All return a score in `[0, 1]` plus structured details. All are deterministic and embedding-free except `judge`.

| name | semantics |
|------|-----------|
| `exact` | string equality after whitespace strip |
| `contains` | case-insensitive substring |
| `regex` | `re.search` with the pattern from config or the case's `expected` |
| `json_subset` | deep subset match `expected ⊆ actual`: extra keys allowed, arrays element-wise, partial credit = fraction of leaf assertions that hold |
| `numeric_tolerance` | `\|a − e\| ≤ max(rel_tol·\|e\|, abs_tol)`; extracts the first number from prose |
| `text_similarity` | token-level F1 (lowercase, punctuation-stripped) — no embeddings |
| `judge` | LLM-as-judge against any OpenAI-compatible endpoint; degrades to a visible `skipped (no endpoint)` result, never a silent zero |

The judge endpoint resolves from `base_url=`, then `OPENAI_BASE_URL` / `LLM_BASE_URL`; the key from `OPENAI_API_KEY`.

## GitHub Action

```yaml
- name: Eval gate
  uses: sam-latino/eval-gate@v0.1.0
  with:
    dataset: examples/golden_dataset.jsonl
    base_url: ${{ secrets.MODEL_BASE_URL }}
    model: gpt-4o
    scorer: exact,json_subset
    k: "3"
    baseline: baseline.json
    github_token: ${{ secrets.GITHUB_TOKEN }}
```

**Action inputs:**

| input | description | default |
|-------|-------------|---------|
| `dataset` | Path to YAML or JSONL eval dataset | required |
| `base_url` | OpenAI-compatible API base URL | `http://localhost:8765` |
| `model` | Model name | `default` |
| `command` | Subprocess command; use `{input}` as placeholder | — |
| `k` | Repeat count (`1` = point-estimate, `>=3` = statistical gate) | `1` |
| `scorer` | Comma-separated scorer list | `contains` |
| `baseline` | Path to baseline JSON | — |
| `threshold` | Pass-rate drop threshold for k=1 gate | `0.05` |
| `out` | Output JSON report path | `evalgate-report.json` |
| `github_token` | Token for sticky PR comment | — |

The action posts a sticky PR comment on every run (updates in place, keeps the PR clean) and writes a GitHub Step Summary.

Note: the action installs `evalgate` via pip during the run. Pin to a release tag for reproducibility.

## Dogfooding

The in-repo echo server and command adapter (`examples/echo_target.py`) exercise the full `baseline → run --k=3 --baseline` statistical-gate round trip in CI. The end-to-end tests in `tests/test_cli_e2e.py` run as part of every push and PR check.

## Development

```bash
pip install -e ".[dev]"
python -m ruff check . && python -m mypy evalgate/ && python -m pytest -q
python scripts/scrub_check.py   # no internal-infra references in the tree
python -m mockserver.server     # local OpenAI-compatible mock
```

## Limitations

- `json_subset` arrays are ordered (element-wise by index); set-like "contains any" matching is not implemented.
- `judge` quality depends entirely on the judge model — treat it as a signal, not ground truth, and keep deterministic scorers as the primary gate.
- The noise floor assumes repeats are exchangeable (same dataset, same decoding parameters). Change either and the committed baseline is stale; regenerate it.
- k=2 is explicitly rejected — the t-interval requires at least 3 degrees of freedom to be meaningful.
- The GitHub Action installs evalgate from PyPI at run time; pin to a release tag to avoid drift in the action itself.

## What I'd do differently

The k>=3 candidate mean is currently the mean of per-repeat means (consistent with the baseline definition). Tracking the full repeat matrix in the candidate report would allow computing a candidate t-interval directly and reporting the degree of overlap — a more principled comparison than point-vs-interval. The data model supports it; the report layer does not yet.

Baseline versioning is single-level (`schema_version: 1`). In practice, teams will want to track model, dataset version, and decoding params alongside the CI stats so a stale baseline is detected rather than silently used.
