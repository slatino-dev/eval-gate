"""End-to-end CLI tests.

Tests cover:
- 'evalgate run' against the in-repo mock HTTP server (both single-run and
  multi-run k-repeat paths).
- 'evalgate run --command' against the self-dogfood echo_target.py.
- 'evalgate baseline --command' writing a valid Baseline JSON.
- 'evalgate run --command --k=3 --baseline' exercising the full statistical gate.
- 'evalgate summary' pretty-printing a saved report.
- Exit-code 1 on regression, exit-code 0 when gate passes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers — mock server fixture
# ──────────────────────────────────────────────────────────────────────────────

MOCK_PORT = 18765  # distinct port so we don't collide with other test runs


@pytest.fixture(scope="module")
def mock_server() -> object:
    """Start the in-repo mock server in a background thread for the module."""
    from mockserver.server import run as _run

    thread = threading.Thread(
        target=_run, kwargs={"port": MOCK_PORT, "verbose": False}, daemon=True
    )
    thread.start()
    # Give the server a moment to bind.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            import urllib.request

            urllib.request.urlopen(f"http://127.0.0.1:{MOCK_PORT}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.05)
    yield thread


def _run_cli(*args: str, expect_code: int = 0) -> subprocess.CompletedProcess[str]:
    """Run 'evalgate <args>' as a subprocess and assert the exit code."""
    result = subprocess.run(
        [sys.executable, "-m", "evalgate.cli"] + list(args),
        capture_output=True,
        text=True,
    )
    assert result.returncode == expect_code, (
        f"Expected exit {expect_code}, got {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dataset paths
# ──────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent
_BASIC_YAML = str(_REPO_ROOT / "examples" / "basic_dataset.yaml")
_CMD_YAML = str(_REPO_ROOT / "examples" / "command_dataset.yaml")
_ECHO_TARGET = str(_REPO_ROOT / "examples" / "echo_target.py")


def _cmd_str(script: str) -> str:
    """Build a --command string that works on Windows paths with spaces.

    On all platforms the path to the script is double-quoted so that
    shlex.split (which CommandAdapter uses) treats it as one token even when
    it contains spaces.
    """
    return f'{sys.executable} "{script}" {{input}}'


# ──────────────────────────────────────────────────────────────────────────────
# run — HTTP adapter (mock server)
# ──────────────────────────────────────────────────────────────────────────────


def test_run_http_exits_zero_no_baseline(mock_server: object) -> None:
    """Single HTTP run, no baseline: gate always passes."""
    result = _run_cli(
        "run",
        _BASIC_YAML,
        f"--base-url=http://127.0.0.1:{MOCK_PORT}",
        "--scorer=contains",
    )
    assert "Results" in result.stdout


def test_run_http_writes_json_report(mock_server: object, tmp_path: Path) -> None:
    """--out writes a valid JSON report."""
    out = tmp_path / "report.json"
    _run_cli(
        "run",
        _BASIC_YAML,
        f"--base-url=http://127.0.0.1:{MOCK_PORT}",
        "--scorer=contains",
        f"--out={out}",
    )
    assert out.is_file()
    data = json.loads(out.read_text())
    assert "summary" in data
    assert data["summary"]["dataset"] == "basic-smoke"
    assert data["summary"]["total"] == 3


def test_run_http_github_summary_flag(mock_server: object) -> None:
    """--github-summary emits the GitHub markdown block."""
    result = _run_cli(
        "run",
        _BASIC_YAML,
        f"--base-url=http://127.0.0.1:{MOCK_PORT}",
        "--github-summary",
    )
    assert "## Eval Gate" in result.stdout


# ──────────────────────────────────────────────────────────────────────────────
# run — command adapter (echo_target.py)
# ──────────────────────────────────────────────────────────────────────────────


def test_run_command_all_pass() -> None:
    """echo_target knows all answers → exact scorer should pass all cases."""
    result = _run_cli(
        "run",
        _CMD_YAML,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--scorer=exact",
    )
    # 4/4 cases in command_dataset.yaml have known answers in echo_target.
    assert "4/4" in result.stdout


def test_run_command_contains_scorer() -> None:
    """contains scorer also passes all cases."""
    result = _run_cli(
        "run",
        _CMD_YAML,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--scorer=contains",
    )
    assert "4/4" in result.stdout


def test_run_command_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "cmd_report.json"
    _run_cli(
        "run",
        _CMD_YAML,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--scorer=exact",
        f"--out={out}",
    )
    data = json.loads(out.read_text())
    assert data["summary"]["passed"] == 4


# ──────────────────────────────────────────────────────────────────────────────
# baseline subcommand
# ──────────────────────────────────────────────────────────────────────────────


def test_baseline_writes_valid_json(tmp_path: Path) -> None:
    bl_path = tmp_path / "baseline.json"
    _run_cli(
        "baseline",
        _CMD_YAML,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--scorer=exact",
        "--k=3",
        f"--out={bl_path}",
    )
    assert bl_path.is_file()
    data = json.loads(bl_path.read_text())
    assert data["schema_version"] == 1
    assert "exact" in data["scorers"]
    stats = data["scorers"]["exact"]
    assert stats["k"] == 3
    assert 0.0 <= stats["mean"] <= 1.0
    assert 0.0 <= stats["ci_low"] <= stats["ci_high"] <= 1.0


def test_baseline_k_below_3_is_rejected(tmp_path: Path) -> None:
    _run_cli(
        "baseline",
        _CMD_YAML,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--k=2",
        expect_code=2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Statistical gate: run --k=3 --baseline
# ──────────────────────────────────────────────────────────────────────────────


def test_statistical_gate_passes_on_same_model(tmp_path: Path) -> None:
    """Building a baseline from echo_target and then running k=3 against the same
    command should always pass — the candidate is the same model, so it stays
    inside the confidence interval."""
    bl_path = tmp_path / "baseline.json"
    out_path = tmp_path / "report.json"

    # Build baseline (k=5 repeats, exact scorer).
    _run_cli(
        "baseline",
        _CMD_YAML,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--scorer=exact",
        "--k=5",
        f"--out={bl_path}",
    )

    # Run k=3 against the same command — should pass.
    result = _run_cli(
        "run",
        _CMD_YAML,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--scorer=exact",
        "--k=3",
        f"--baseline={bl_path}",
        f"--out={out_path}",
        expect_code=0,
    )
    # Drift section in text output.
    assert "Gate" in result.stdout

    # JSON report should have a 'drift' key (DriftResult, not DriftReport).
    data = json.loads(out_path.read_text())
    assert "drift" in data
    assert not data["drift"]["regressed"]


def test_statistical_gate_regression_exits_1(tmp_path: Path) -> None:
    """When the baseline has a mean of 1.0 (zero variance) but the candidate
    runs a command that always returns an empty string (score=0), the gate
    must fail and exit 1."""
    from evalgate.drift import Baseline, compute_stats

    # Build a synthetic perfect baseline (mean=1.0, zero variance) and save it.
    bl = Baseline(
        scorers={"exact": compute_stats("exact", [1.0, 1.0, 1.0])},
    )
    bl_path = tmp_path / "perfect_baseline.json"
    bl.save(bl_path)

    # A command that always outputs nothing → score=0 for exact.
    empty_cmd = f"{sys.executable} -c \"import sys; print('')\"  {{input}}"
    # (empty string vs expected 'Paris' etc. → exact match fails)
    _run_cli(
        "run",
        _CMD_YAML,
        f"--command={empty_cmd}",
        "--scorer=exact",
        "--k=3",
        f"--baseline={bl_path}",
        expect_code=1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# summary subcommand
# ──────────────────────────────────────────────────────────────────────────────


def test_summary_pretty_prints_saved_report(tmp_path: Path) -> None:
    out = tmp_path / "rep.json"
    _run_cli(
        "run",
        _CMD_YAML,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--scorer=exact",
        f"--out={out}",
    )
    result = _run_cli("summary", str(out))
    assert "Dataset" in result.stdout
    assert "Results" in result.stdout


def test_summary_github_flag(tmp_path: Path) -> None:
    out = tmp_path / "rep2.json"
    _run_cli(
        "run",
        _CMD_YAML,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--scorer=exact",
        f"--out={out}",
    )
    result = _run_cli("summary", str(out), "--github")
    assert "## Eval Gate" in result.stdout


# ──────────────────────────────────────────────────────────────────────────────
# k=2 is rejected
# ──────────────────────────────────────────────────────────────────────────────


def test_run_k2_is_rejected() -> None:
    _run_cli(
        "run",
        _CMD_YAML,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--k=2",
        expect_code=2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Unknown scorer is rejected cleanly
# ──────────────────────────────────────────────────────────────────────────────


def test_run_unknown_scorer_exits_2(mock_server: object) -> None:
    _run_cli(
        "run",
        _BASIC_YAML,
        f"--base-url=http://127.0.0.1:{MOCK_PORT}",
        "--scorer=nonexistent",
        expect_code=2,
    )


# ──────────────────────────────────────────────────────────────────────────────
# JSONL dataset dispatch
# ──────────────────────────────────────────────────────────────────────────────

_CMD_JSONL = str(_REPO_ROOT / "examples" / "command_dataset.jsonl")


def test_run_jsonl_dataset_dispatch() -> None:
    """evalgate run accepts a .jsonl dataset and produces correct results."""
    result = _run_cli(
        "run",
        _CMD_JSONL,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--scorer=exact",
    )
    # All 4 cases known to echo_target should pass.
    assert "4/4" in result.stdout


def test_baseline_jsonl_dataset_dispatch(tmp_path: Path) -> None:
    """evalgate baseline also accepts a .jsonl dataset."""
    bl_path = tmp_path / "baseline_jsonl.json"
    _run_cli(
        "baseline",
        _CMD_JSONL,
        f"--command={_cmd_str(_ECHO_TARGET)}",
        "--scorer=exact",
        "--k=3",
        f"--out={bl_path}",
    )
    assert bl_path.is_file()
