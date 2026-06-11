"""Trivial command target for the evalgate self-dogfood example.

Usage (from the repo root):
    python examples/echo_target.py "what is the capital of France?"

Outputs a deterministic answer for a small fixed set of questions so the
examples/command_dataset.yaml golden set can be evaluated with perfect scores
without any network call.  Unknown inputs are echoed back verbatim so the
command always exits 0.
"""

from __future__ import annotations

import sys

_ANSWERS: dict[str, str] = {
    "what is 2 + 2?": "4",
    "what is the capital of france?": "Paris",
    "say hello.": "hello",
    "is the sky blue?": "yes",
}


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: echo_target.py <question>")
    question = " ".join(sys.argv[1:]).strip()
    print(_ANSWERS.get(question.lower(), question))


if __name__ == "__main__":
    main()
