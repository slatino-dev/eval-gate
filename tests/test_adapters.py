"""Tests for CommandAdapter and OpenAIAdapter."""

from __future__ import annotations

import sys

import pytest

from evalgate.adapters import CommandAdapter, CommandError

# ──────────────────────────────────────────────────────────────────────────────
# CommandAdapter
# ──────────────────────────────────────────────────────────────────────────────


def test_command_adapter_substitutes_input() -> None:
    """The {input} placeholder is replaced with the actual input value."""
    # Use a Python one-liner that prints its argument so we're not relying on
    # any system tool.  We build the command as a list so no shell is invoked.
    adapter = CommandAdapter(
        cmd=[sys.executable, "-c", "import sys; print(sys.argv[1])", "{input}"]
    )
    out = adapter.run("hello-world")
    assert out.strip() == "hello-world"


def test_command_adapter_string_template_splits_correctly() -> None:
    """A string template is split by shlex before any {input} substitution."""
    # shlex.split happens at construction time; the {input} is only replaced at
    # run time — so spaces inside the input value don't split the argument list.
    adapter = CommandAdapter(
        cmd=f"{sys.executable} -c \"import sys; print(sys.argv[1])\" {{input}}"
    )
    out = adapter.run("a b c")  # spaces in input — must stay one token
    assert out.strip() == "a b c"


def test_command_adapter_nonzero_exit_raises_command_error() -> None:
    adapter = CommandAdapter(cmd=[sys.executable, "-c", "import sys; sys.exit(1)"])
    with pytest.raises(CommandError) as exc_info:
        adapter.run("ignored")
    assert exc_info.value.returncode == 1


def test_command_error_str_includes_returncode() -> None:
    err = CommandError(returncode=42, stderr="something went wrong")
    assert "42" in str(err)


def test_command_adapter_run_batch_collects_errors() -> None:
    """run_batch returns a mix of str and CommandError in order."""
    adapter = CommandAdapter(
        cmd=[
            sys.executable,
            "-c",
            # Exit non-zero when the input is "fail", otherwise print it.
            "import sys; inp=sys.argv[1]; sys.exit(1) if inp=='fail' else print(inp)",
            "{input}",
        ]
    )
    results = adapter.run_batch(["ok", "fail", "also-ok"])
    assert isinstance(results[0], str) and results[0].strip() == "ok"
    assert isinstance(results[1], CommandError) and results[1].returncode == 1
    assert isinstance(results[2], str) and results[2].strip() == "also-ok"


def test_command_adapter_shell_false_metacharacters_are_literal() -> None:
    """Shell metacharacters in the input reach the program verbatim."""
    adapter = CommandAdapter(
        cmd=[sys.executable, "-c", "import sys; print(repr(sys.argv[1]))", "{input}"]
    )
    # If shell=True were used, ; would start a new command and rm -rf / could run.
    dangerous_input = "$(rm -rf /); `evil`"
    out = adapter.run(dangerous_input)
    # The program receives the string literally and repr()s it; it should NOT
    # interpret the metacharacters.
    assert "rm" in out and "evil" in out  # present as literal text


def test_command_adapter_stdout_captured_stderr_not_mixed() -> None:
    """Only stdout is returned; stderr is NOT included in the result string."""
    adapter = CommandAdapter(
        cmd=[
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('err-line\\n'); print('stdout-line')",
        ]
    )
    out = adapter.run("unused")
    assert "stdout-line" in out
    assert "err-line" not in out
