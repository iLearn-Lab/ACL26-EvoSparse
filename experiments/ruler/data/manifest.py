"""Minimal JSONL manifest helpers used by the RULER pipeline."""

from __future__ import annotations

import json
from pathlib import Path


def read_manifest(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_manifest(path, records):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
