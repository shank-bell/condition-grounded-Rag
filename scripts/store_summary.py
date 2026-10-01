"""Print a short summary of the profile store: profiles per paper, chunks with profiles (no LLM needed)."""
from __future__ import annotations

import sqlite3

from cgrag.config import get_settings


def main() -> None:
    db = sqlite3.connect(get_settings().paths.profile_db)
    rows = db.execute("SELECT paper_id, COUNT(*) FROM profiles GROUP BY paper_id ORDER BY paper_id").fetchall()
    print(f"{len(rows)} papers, {sum(n for _, n in rows)} profiles, "
          f"{db.execute('SELECT COUNT(DISTINCT chunk_id) FROM profiles').fetchone()[0]} chunks with profiles")
    print(" ".join(f"{p}:{n}" for p, n in rows))


if __name__ == "__main__":
    main()
