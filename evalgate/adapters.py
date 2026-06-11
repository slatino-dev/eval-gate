"""Adapters for calling systems under test.

Two adapters are provided:

- :class:`OpenAIAdapter` — thin wrapper around OpenAI-compatible
  ``/v1/chat/completions`` endpoints; used for LLM eval.
- :class:`CommandAdapter` — spawns a subprocess per case (or in batch)
  with ``subprocess.run(..., shell=False)``; used for CLI / agent evals.
  Arguments are always passed as a list so shell metacharacters in any
  input value cannot be interpreted.
"""

from __future__ import annotations

import subprocess
from typing import Any

import httpx
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible HTTP adapter
# ──────────────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int = 512


class CompletionResponse(BaseModel):
    id: str = ""
    model: str = ""
    content: str = ""
    usage: dict[str, int] = Field(default_factory=dict)


class OpenAIAdapter:
    """Thin adapter for OpenAI-compatible /v1/chat/completions endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "not-needed",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a chat completion request and return the parsed response."""
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = ""
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")

        return CompletionResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            content=content,
            usage=data.get("usage", {}),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Command (subprocess) adapter
# ──────────────────────────────────────────────────────────────────────────────


class CommandError(RuntimeError):
    """Raised when a command exits with a non-zero status."""

    def __init__(self, returncode: int, stderr: str) -> None:
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"command exited {returncode}: {stderr[:500]}")


class CommandAdapter:
    """Invoke an external command per eval case and capture stdout as the response.

    The command template uses ``{input}`` as a placeholder for the case's
    input value.  At call time the placeholder is replaced with the actual
    input string and the resulting tokens are split into a list and passed
    directly to ``subprocess.run``.  **``shell=False`` is always used** so no
    shell is spawned and metacharacters in the input value cannot be
    interpreted.

    Example::

        adapter = CommandAdapter(cmd=["my-tool", "--query", "{input}"])
        output = adapter.run("what is 2+2?")   # → stdout of my-tool

    Alternatively, pass ``cmd`` as a string containing ``{input}`` and the
    adapter will ``shlex.split`` it at construction time (safe: splitting
    happens before any user input is substituted)::

        adapter = CommandAdapter(cmd="my-tool --query {input}")

    ``timeout`` (seconds) is applied per subprocess call; ``CommandError`` is
    raised on non-zero exit.
    """

    def __init__(
        self,
        cmd: list[str] | str,
        timeout: float = 30.0,
        encoding: str = "utf-8",
    ) -> None:
        import os
        import shlex

        if isinstance(cmd, str):
            # On POSIX, shlex.split(posix=True) handles backslash escaping as
            # escape characters.  On Windows, paths use backslashes and the POSIX
            # rules would eat them — use posix=False so ``C:\foo\bar`` is
            # preserved.  In non-POSIX mode shlex keeps surrounding quote
            # characters as part of the token, so we strip them afterward.
            posix = os.name != "nt"
            tokens = shlex.split(cmd, posix=posix)
            if not posix:
                # Strip a single layer of surrounding double or single quotes
                # that shlex preserved in non-posix mode.
                self._template: list[str] = [
                    t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'") else t
                    for t in tokens
                ]
            else:
                self._template = tokens
        else:
            self._template = list(cmd)
        self.timeout = timeout
        self.encoding = encoding

    def run(self, input_value: str) -> str:
        """Run the command with ``input_value`` substituted and return stdout.

        Raises :class:`CommandError` if the process exits non-zero.
        """
        args = [token.replace("{input}", input_value) for token in self._template]
        result = subprocess.run(  # noqa: S603 — shell=False, args are a list
            args,
            shell=False,
            capture_output=True,
            timeout=self.timeout,
            encoding=self.encoding,
        )
        if result.returncode != 0:
            raise CommandError(result.returncode, result.stderr)
        return result.stdout

    def run_batch(self, inputs: list[str]) -> list[str | CommandError]:
        """Run one subprocess per input; returns results or errors in order."""
        outputs: list[str | CommandError] = []
        for inp in inputs:
            try:
                outputs.append(self.run(inp))
            except CommandError as exc:
                outputs.append(exc)
        return outputs
