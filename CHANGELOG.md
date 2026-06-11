# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-06-11

### Added

- **Statistical noise-floor gate** (`evalgate/drift.py`): Student-t confidence interval over k>=3 repeat means; flags regressions only when the candidate falls *below* the baseline CI, ignoring within-noise fluctuations. t critical values computed via the regularised incomplete beta function (no scipy); unit-tested against published t-tables.
- **Golden dataset schema** (`evalgate/dataset.py`): pydantic `EvalCase` + `EvalDataset` with `extra="forbid"` for loud-fail on typos, line-precise errors in JSONL, and both `.jsonl` and `.yaml` loaders. CLI dispatches on file extension.
- **Seven scorers** (`evalgate/scorers.py`): `exact`, `contains`, `regex`, `json_subset` (deep subset with partial credit), `numeric_tolerance` (prose-safe number extraction), `text_similarity` (token F1, no embeddings), `judge` (LLM-as-judge, degrades gracefully to `skipped` when no endpoint is configured).
- **Three CLI subcommands** (`evalgate/cli.py`): `run` (single or k-repeat), `baseline` (build noise-floor baseline), `summary` (pretty-print saved report).
- **Adapters** (`evalgate/adapters.py`): `OpenAIAdapter` (httpx, any OpenAI-compatible endpoint) and `CommandAdapter` (subprocess with `shell=False`, injection-safe).
- **Report layer** (`evalgate/report.py`): plain-text summary, GitHub Step Summary markdown, and machine-readable JSON report with per-scorer drift table.
- **GitHub Action** (`action/`): composite action (installs evalgate, runs CLI, posts sticky PR comment); exposes `dataset`, `k`, `scorer`, `command`, `baseline`, `threshold`, `out`, and `github_token` inputs.
- **Mock server** (`mockserver/`): local OpenAI-compatible server for offline testing and CI.
- **End-to-end tests** (`tests/test_cli_e2e.py`): full `baseline → run --k=3 --baseline` statistical-gate round trip against the in-repo echo target; exit-code assertions for pass/regression/usage-error.
- **Scrub gate** (`scripts/scrub_check.py`): CI job that blocks internal-infra references (hostnames, CGNAT ranges, API-key patterns) before anything ships publicly.

### Fixed

- Report delta rendering: positive deltas now render as `+0.140` instead of `++0.140` (the `_sign()` helper and the `:+.3f` format spec were both emitting a sign character in the Python CLI output).
- `_single_run_summary` no longer calls the adapter twice per case — pass/fail and scores now derive from the same generation, halving API cost for single-run evaluations.
- All-skipped scorer repeats now emit a visible warning and are excluded from the candidate mean rather than silently substituting `0.0`.
- k>=3 candidate mean is now the mean of per-repeat means (consistent with how `summarize_run` defines the baseline mean) rather than a pool of all case scores across repeats.
- CI workflow branch trigger now includes `master` so the existing branch is covered.
- `action/dist/` removed from `.gitignore` so the bundled JS is tracked and the action works at point of use.
- Phantom `numpy>=1.26` dependency removed from `pyproject.toml` (numpy is not imported anywhere in the package).

[0.1.0]: https://github.com/sam-latino/eval-gate/releases/tag/v0.1.0
