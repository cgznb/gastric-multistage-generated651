"""Atomic bindings, source fingerprints and space-efficient checkpoint pointers."""

import hashlib
import json
import os
from pathlib import Path

from stageworld.artifacts import atomic_write_private_json, read_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_hash(root: Path) -> str:
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    return object_hash({str(p.relative_to(root)): sha256(p) for p in files})


def bound_json(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing experiment binding differs: {path.name}")
    else:
        atomic_write_private_json(path, value)


def checkpoint_link(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.link")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, target)
