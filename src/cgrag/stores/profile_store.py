"""Condition Profile Store: SQLite, keyed by chunk_id. Read by stage 6 (with the question) and stage 7 (pairwise)."""
from __future__ import annotations

import json
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
-- the repair overlay (ingestion/repair.py): only the fields a rule changed, per profile; `profiles` itself is never rewritten
CREATE TABLE IF NOT EXISTS profile_repairs (
    profile_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, fields TEXT NOT NULL, rules TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_repairs_paper ON profile_repairs(paper_id);
"""


class ProfileStore:
    def __init__(self, path: Path | None = None, repaired: bool | None = None) -> None:
        """`repaired`: read profiles through the repair overlay (None = `[features] profile_repair`); False gives the raw extraction (the evaluation
        of the repair itself, and the ablation, read the store this way)."""
        path = path or get_settings().paths.profile_db
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._cache: list[ConditionProfile] | None = None          # every profile, for chunks_recording()
        with self.lock:
            self.db.executescript(_SCHEMA)
        self.repaired = get_settings().features.profile_repair if repaired is None else repaired
        self._overrides: dict[str, dict] = self._load_overrides() if self.repaired else {}

    # ---- the repair overlay -------------------------------------------------------------------------------------------------------------------
    def _load_overrides(self) -> dict[str, dict]:
        with self.lock:
            return {r[0]: json.loads(r[1]) for r in self.db.execute("SELECT profile_id, fields FROM profile_repairs")}

    def set_repairs(self, paper_id: str, raw: list[ConditionProfile], fixed: list[ConditionProfile], rules: dict[str, list[str]] | None = None) -> int:
        """Store, for one paper, the fields in which `fixed[i]` differs from `raw[i]` (replaces the paper's earlier overlay). Returns the number of
        repaired profiles."""
        rows = []
        for a, b in zip(raw, fixed):
            diff = {c: getattr(b, c) for c in _COLS if c not in ("profile_id", "paper_id", "chunk_id", "methods_chunk_id", "evidence")
                    and (getattr(a, c) or None) != (getattr(b, c) or None)}
            if diff:
                rows.append((a.profile_id, paper_id, json.dumps(diff, ensure_ascii=False), json.dumps((rules or {}).get(a.profile_id, []))))
        with self.lock, self.db:
            self.db.execute("DELETE FROM profile_repairs WHERE paper_id=?", (paper_id,))
            self.db.executemany("INSERT OR REPLACE INTO profile_repairs VALUES (?,?,?,?)", rows)
        if self.repaired:
            self._overrides = self._load_overrides()
        self._cache = None
        return len(rows)

    def clear_repairs(self, paper_id: str | None = None) -> None:
        with self.lock, self.db:
            if paper_id is None:
                self.db.execute("DELETE FROM profile_repairs")
            else:
                self.db.execute("DELETE FROM profile_repairs WHERE paper_id=?", (paper_id,))
        self._overrides = self._load_overrides() if self.repaired else {}
        self._cache = None

    def repair_count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM profile_repairs").fetchone()[0]

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
            self.db.execute("DELETE FROM profile_repairs WHERE paper_id=?", (paper_id,))
            self._cache = None
        self._overrides = {k: v for k, v in self._overrides.items() if not k.startswith(f"{paper_id}:")}

    def _row(self, r: sqlite3.Row) -> ConditionProfile:
        p = ConditionProfile(**{c: r[c] for c in _COLS})
        ov = self._overrides.get(p.profile_id)
        return p.model_copy(update=ov) if ov else p

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

    def _cached(self) -> list[ConditionProfile]:
        """Every profile, loaded once (the cache is dropped when profiles are added or a paper is deleted)."""
        with self.lock:
            if self._cache is None:
                self._cache = [self._row(r) for r in self.db.execute("SELECT * FROM profiles")]
            return self._cache

    def profiles_matching(self, requested: dict[str, str], fields: list[str] | tuple[str, ...]) -> list[ConditionProfile]:
        """Profiles that record ALL of the given conditions together: for each field in `fields` (taken from `requested`) a value
        that satisfies it (`values_match`: 'mBERT' does not satisfy 'BERT', 'kn' satisfies 'Kannada'). Used by Stage 6's joint
        coverage check: three conditions can each be covered by a different paper without any paper covering them together."""
        from ..pipeline.conditions import covers, observed_values
        wanted = {f: requested[f] for f in fields if requested.get(f)}
        if not wanted:
            return []
        return [p for p in self._cached()
                if all(any(covers(f, w, v) for v in observed_values(p, f)) for f, w in wanted.items())]

    def chunks_recording(self, requested: dict[str, str], missing: str, limit: int = 4) -> list[str]:
        """Chunks whose profiles record the `missing` condition, best first (Stage 6, profile-guided retrieval).

        Only a chunk with ONE profile that records the missing condition AND every other requested condition counts:
        for {language: Kannada, task: NLI} that is a Kannada NLI result, not a Kannada QA result and not an NLI result in
        Hindi; for {language: Kannada, dataset: XNLI} nothing qualifies (XNLI has no Kannada), so no chunk is added and the
        scope warning stays honest. Ties are ordered by how many matching profiles the chunk holds."""
        from ..pipeline.conditions import covers, observed_values
        wanted = requested.get(missing)
        if not wanted:
            return []
        scores: dict[str, list[int]] = {}                       # chunk_id -> [best score, matching profiles]
        for p in self._cached():
            if not any(covers(missing, wanted, v) for v in observed_values(p, missing)):
                continue
            s = 1 + sum(1 for f, w in requested.items()
                        if f != missing and any(covers(f, w, v) for v in observed_values(p, f)))
            entry = scores.setdefault(p.chunk_id, [0, 0])
            entry[0], entry[1] = max(entry[0], s), entry[1] + 1
        every = len(requested)                                   # the missing condition plus all the others
        ranked = sorted((cid for cid, (s, _) in scores.items() if s == every), key=lambda cid: -scores[cid][1])
        return ranked[:limit]

    def vocabulary(self) -> dict[str, list[str]]:
        """Distinct values recorded per condition field: the names the paper store already knows."""
        from ..pipeline.conditions import is_nullish
        out: dict[str, list[str]] = {}
        profiles = self._cached()                                   # through the repair overlay, like every other read
        for field in ("dataset", "model", "language", "task"):
            values = {getattr(p, field) for p in profiles if getattr(p, field) is not None}
            out[field] = sorted(v for v in values if not is_nullish(v))
        return out

    def paper_count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM extraction_log").fetchone()[0]

    def count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
