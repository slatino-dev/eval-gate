# examples/

Self-contained datasets and helpers for running evalgate locally and in tests.

## Files

| File | Purpose |
|------|---------|
| `basic_dataset.yaml` | Three-case smoke dataset used by CI against the mock server (HTTP adapter path). |
| `command_dataset.yaml` | Four-case YAML dataset for the command adapter (`--command`) path; answers are all known to `echo_target.py`. |
| `command_dataset.jsonl` | JSONL version of `command_dataset.yaml` — exercises the `.jsonl` file-extension dispatch in the CLI. |
| `golden_dataset.jsonl` | Richer five-case JSONL set showing all five deterministic scorers (exact, json_subset, numeric_tolerance, regex, text_similarity) plus one judge case. Canonical format reference. |
| `echo_target.py` | Deterministic command target that answers the four questions in `command_dataset.*` without a network call. Used by the end-to-end CLI tests. |

## Quickstart (command adapter, no API key needed)

```bash
# build a k=5 noise-floor baseline
evalgate baseline examples/command_dataset.yaml \
    --command="python examples/echo_target.py {input}" \
    --scorer=exact --k=5 --out=baseline.json

# check a candidate — passes because it is the same command
evalgate run examples/command_dataset.yaml \
    --command="python examples/echo_target.py {input}" \
    --scorer=exact --k=3 --baseline=baseline.json
```

## Quickstart (HTTP adapter, mock server)

```bash
# start the included mock OpenAI-compatible server
python -m mockserver.server

# single-run check
evalgate run examples/basic_dataset.yaml \
    --base-url=http://localhost:8765 --scorer=contains
```
