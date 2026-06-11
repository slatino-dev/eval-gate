#!/usr/bin/env python3
"""scrub_check.py - fail (exit 1) if any tracked text file contains a likely
secret (private keys, cloud / API tokens) or a carrier-grade-NAT address.

A lightweight pre-commit / CI guard. Generic by design: it ships no project- or
host-specific values. Usage: python scripts/scrub_check.py [--path DIR]
"""
from __future__ import annotations
import re, subprocess, sys
from pathlib import Path

PATTERNS = [
    ("private-key-block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("aws-access-key-id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b")),
    ("openai-style-key", re.compile(r"\bsk-(?:ant-|proj-|or-v1-|live-)?[A-Za-z0-9]{32,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{12,}\b")),
    ("cgnat-address", re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")),
]
SKIP_SUFFIX = {".png",".jpg",".jpeg",".gif",".ico",".svg",".pdf",".whl",".pyc",".so",".dll",".exe",".lock",".bin",".gz",".zip",".tar",".woff",".woff2",".ttf",".otf"}
SKIP_PARTS = {".git","node_modules","target","dist","build","__pycache__",".venv",".mypy_cache",".ruff_cache",".astro","vendor"}

def tracked(root: Path):
    try:
        out = subprocess.run(["git","-C",str(root),"ls-files"],capture_output=True,text=True,check=True).stdout
        return [root / ln for ln in out.splitlines() if ln]
    except Exception:
        return [p for p in root.rglob("*") if p.is_file()]

def main() -> int:
    args = sys.argv[1:]
    root = Path(args[args.index("--path")+1]).resolve() if "--path" in args else Path(".").resolve()
    self_path = Path(__file__).resolve()
    found = 0
    for f in tracked(root):
        if f.resolve() == self_path or f.suffix.lower() in SKIP_SUFFIX: continue
        if any(part in SKIP_PARTS for part in f.parts) or "min." in f.name: continue
        try: text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception: continue
        for label, rx in PATTERNS:
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    print(f"SCRUB FAIL [{label}] {f}:{i}"); found += 1
    if found:
        print(f"scrub_check: FAILED - {found} match(es)."); return 1
    print("scrub_check: OK - no secrets detected."); return 0

if __name__ == "__main__":
    raise SystemExit(main())
