"""Regenerate the release manifest with deterministic paths and SHA-256 hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__"}
EXCLUDED_NAMES = {"MANIFEST.json"}


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and not any(part in EXCLUDED_PARTS for part in relative.parts)
        and path.name not in EXCLUDED_NAMES
        and path.suffix != ".pyc"
    )


def main():
    files = sorted((p for p in ROOT.rglob("*") if included(p)), key=lambda p: p.relative_to(ROOT).as_posix())
    entries = []
    for path in files:
        payload = path.read_bytes()
        entries.append({
            "path": path.relative_to(ROOT).as_posix(),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest = {
        "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
        "file_count": len(entries), "files": entries,
    }
    (ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Manifest v{manifest['version']}: {len(entries)} files")


if __name__ == "__main__":
    main()
