"""Condition Profile Store: SQLite, keyed by chunk_id. Read by stage 6 (with the question) and stage 7 (pairwise)."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from ..config import get_settings
from ..schemas import ConditionProfile

_COLS = ("profile_id", "paper_id", "chunk_id", "methods_chunk_id", "task", "dataset", "dataset_version",
         "metric", "value", "language", "model", "model_size", "setting", "evidence")

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS profiles (
    profile_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, chunk_id TEXT NOT NULL, methods_chunk_id TEXT,
    task TEXT, dataset TEXT, dataset_version TEXT, metric TEXT, value REAL, language TEXT,
    model TEXT, model_size TEXT, setting TEXT, evidence TEXT
);
CREATE INDEX IF NOT EXISTS idx_profiles_chunk ON profiles(chunk_id);
CREATE INDEX IF NOT EXISTS idx_profiles_paper ON profiles(paper_id);
CREATE TABLE IF NOT EXISTS extraction_log (
    paper_id TEXT PRIMARY KEY, chunks_processed INTEGER, profiles INTEGER, llm TEXT, seconds REAL, ts REAL
);
"""


class ProfileStore:
    def __init__(self, path: Path | None = None) -> None:
        path = path or get_settings().paths.profile_db
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._cache: list[ConditionProfile] | None = None          # every profile, for chunks_recording()
        with self.lock:
            self.db.executescript(_SCHEMA)

    def add_many(self, profiles: list[ConditionProfile]) -> None:
        rows = [tuple(getattr(p, c) for c in _COLS) for p in profiles]
        with self.lock, self.db:
            self.db.executemany(
                f"INSERT OR REPLACE INTO profiles ({','.join(_COLS)}) VALUES ({','.join('?' * len(_COLS))})", rows)
            self._cache = None

    def log_extraction(self, paper_id: str, chunks_processed: int, n_profiles: int, llm: str, seconds: float) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO extraction_log VALUES (?,?,?,?,?,?)",
                            (paper_id, chunks_processed, n_profiles, llm, seconds, time.time()))

    def was_extracted(self, paper_id: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM extraction_log WHERE paper_id=?", (paper_id,)).fetchone() is not None

    def delete_paper(self, paper_id: str) -> None:
        with self.lock, self.db:
            self.db.execute("DELETE FROM profiles WHERE paper_id=?", (paper_id,))
            self.db.execute("DELETE FROM extraction_log WHERE paper_id=?", (paper_id,))
            self._cache = None

    @staticmethod
    def _row(r: sqlite3.Row) -> ConditionProfile:
        return ConditionProfile(**{c: r[c] for c in _COLS})

    def for_chunks(self, chunk_ids: list[str]) -> dict[str, list[ConditionProfile]]:
        """Profiles grouped by chunk_id (chunks without any profile are absent)."""
        out: dict[str, list[ConditionProfile]] = {}
        if not chunk_ids:
            return out
        with self.lock:
            for i in range(0, len(chunk_ids), 500):
                part = chunk_ids[i:i + 500]
                q = f"SELECT * FROM profiles WHERE chunk_id IN ({','.join('?' * len(part))})"
                for r in self.db.execute(q, part):
                    out.setdefault(r["chunk_id"], []).append(self._row(r))
        return out

    def for_paper(self, paper_id: str) -> list[ConditionProfile]:
        with self.lock:
            return [self._row(r) for r in self.db.execute("SELECT * FROM profiles WHERE paper_id=? ORDER BY profile_id", (paper_id,))]

    def all(self) -> list[ConditionProfile]:
        with self.lock:
            return [self._row(r) for r in self.db.execute("SELECT * FROM profiles ORDER BY profile_id")]

    def chunks_recording(self, requested: dict[str, str], missing: str, limit: int = 4) -> list[str]:
        """Chunks whose profiles record the `missing` condition, best first (Stage 6, profile-guided retrieval).

        Only a chunk with ONE profile that records the missing condition AND every other requested condition counts:
        for {language: Kannada, task: NLI} that is a Kannada NLI result, not a Kannada QA result and not an NLI result in
        Hindi; for {language: Kannada, dataset: XNLI} nothing qualifies (XNLI has no Kannada), so no chunk is added and the
        scope warning stays honest. Ties are ordered by how many matching profiles the chunk holds."""
        from ..pipeline.conditions import observed_values, values_match
        wanted = requested.get(missing)
        if not wanted:
            return []
        with self.lock:
            if self._cache is None:
                self._cache = [self._row(r) for r in self.db.execute("SELECT * FROM profiles")]
            profiles = self._cache
        scores: dict[str, list[int]] = {}                       # chunk_id -> [best score, matching profiles]
        for p in profiles:
            if not any(values_match(missing, wanted, v) for v in observed_values(p, missing)):
                continue
            s = 1 + sum(1 for f, w in requested.items()
                        if f != missing and any(values_match(f, w, v) for v in observed_values(p, f)))
            entry = scores.setdefault(p.chunk_id, [0, 0])
            entry[0], entry[1] = max(entry[0], s), entry[1] + 1
        every = len(requested)                                   # the missing condition plus all the others
        ranked = sorted((cid for cid, (s, _) in scores.items() if s == every), key=lambda cid: -scores[cid][1])
        return ranked[:limit]

    def vocabulary(self) -> dict[str, list[str]]:
        """Distinct values recorded per condition field: the names the paper store already knows."""
        from ..pipeline.conditions import is_nullish
        out: dict[str, list[str]] = {}
        with self.lock:
            for field in ("dataset", "model", "language", "task"):
                rows = self.db.execute(f"SELECT DISTINCT {field} FROM profiles WHERE {field} IS NOT NULL").fetchall()
                out[field] = [r[0] for r in rows if not is_nullish(r[0])]
        return out

    def paper_count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM extraction_log").fetchone()[0]

    def count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
