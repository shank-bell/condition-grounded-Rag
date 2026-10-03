"""The record that makes a result freezable: which commit, which uncommitted files and when. Every evaluation output carries it, so the
paper can say exactly what produced a number - and that the system's outputs were fixed before the team's labels were known."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def provenance() -> dict:
    try:
        head, modified = _git("rev-parse", "HEAD").strip(), _git("status", "--porcelain").splitlines()
    except Exception:
        head, modified = "unknown", ["unknown"]
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frozen_at_commit": head,
            "modified_files_at_that_time": modified}
