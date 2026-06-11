#!/usr/bin/env python3
"""
scrub_check.py — fail if the working tree leaks internal infra references.

Exits 1 and prints matching file/line if any of these patterns are found:
  - 'redacted-host'
  - 100.64.x CGNAT address (e.g. redacted-ip)
  - 'redacted-domain'
  - 'redacted-mesh'
  - 'redacted-key'
  - generic API key: sk-[A-Za-z0-9]{20,}
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("redacted-host hostname", re.compile(r"redacted-host", re.IGNORECASE)),
    ("CGNAT 100.64.x address", re.compile(r"\b100\.64\.\d{1,3}\.\d{1,3}\b")),
    ("redacted-domain domain", re.compile(r"redacted-domain", re.IGNORECASE)),
    ("redacted-mesh reference", re.compile(r"redacted-mesh", re.IGNORECASE)),
    ("redacted-key key", re.compile(r"redacted-key", re.IGNORECASE)),
    ("generic API key sk-*", re.compile(r"sk-[A-Za-z0-9]{20,}")),
]

# Directories / file patterns to skip entirely.
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "dist", ".mypy_cache"}
SKIP_FILES = {"scrub_check.py"}  # don't scan ourselves for the patterns

TEXT_EXTENSIONS = {
    ".py", ".ts", ".js", ".mjs", ".cjs",
    ".yaml", ".yml", ".toml", ".json", ".md",
    ".txt", ".sh", ".env", ".cfg", ".ini",
}


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS and path.suffix != "":
            continue
        yield path


def main() -> int:
    root = Path(__file__).parent.parent
    findings: list[str] = []

    for filepath in sorted(iter_files(root)):
        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    rel = filepath.relative_to(root)
                    findings.append(f"  [{label}] {rel}:{lineno}: {line.strip()}")

    if findings:
        print("scrub_check FAILED — internal infra references detected:\n")
        for f in findings:
            print(f)
        print(f"\n{len(findings)} finding(s). Remove before committing.")
        return 1

    print("scrub_check passed — no internal infra references found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
