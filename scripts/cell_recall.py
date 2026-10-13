"""Share of the decimal result cells of the recognised result tables that have a stored profile with the same number in the same chunk, and the worst chunks.

  python scripts/cell_recall.py [--min-missing 6] [--out data/index/_logs/low_recall_chunks.txt]

Statistics tables (by caption) are skipped; a garbled table is counted through its rescued blocks, or skipped when none can be split. The chunks with at least
`--min-missing` cells without a profile are written to `--out` (space separated): the list for `rescue_garbled.py --apply --complete`. Reads the profile store RAW (no
overlay) and data/index/chunks.jsonl; with CGRAG_OVERLAY it reads the benchmark store instead. Nothing here needs Ollama.
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgrag.config import get_settings
from cgrag.ingestion.tables import (GARBLE_LIMIT, garble_ratio, is_statistics_table, is_structured_table, parse_table, rescue_tables,
                                    table_cells)
from cgrag.labelling.sampling import load_chunk_index
from cgrag.stores.profile_store import ProfileStore

ap = argparse.ArgumentParser()
ap.add_argument("--min-missing", type=int, default=6)
ap.add_argument("--out", type=Path, default=ROOT / "data/index/_logs/low_recall_chunks.txt")
args = ap.parse_args()

idx = load_chunk_index(get_settings().paths.index_dir / "chunks.jsonl")
st = ProfileStore(repaired=False)
vals = defaultdict(set)
for p in st.all():
    if p.value is not None:
        vals[p.chunk_id].add(round(float(p.value), 4))
NUM = re.compile(r"(?<![\d.])\d+\.\d+(?![\d])")
tot = hit = 0
rows = []
skipped_garbled = skipped_stats = 0
for cid, text in sorted(idx.texts.items()):
    if not is_structured_table(text):
        continue
    t = parse_table(text)
    if is_statistics_table(t.caption):
        skipped_stats += 1
        continue
    pieces = [text]
    if garble_ratio(text) >= GARBLE_LIMIT:
        pieces = rescue_tables(text) or []
        if not pieces:
            skipped_garbled += 1
            continue
    cells = []
    for piece in pieces:
        tt = parse_table(piece)
        for c in table_cells(tt):
            for n in NUM.findall(c.text):
                cells.append((c, float(n)))
    if not cells:
        continue
    got = sum(1 for c, n in cells if round(n, 4) in vals.get(cid, ()))
    tot += len(cells); hit += got
    rows.append((len(cells) - got, len(cells), cid, idx.pages[cid], t.caption[:70]))
print(f"result tables: {len(rows)}; decimal cells {tot}; with a profile {hit} = {100 * hit / tot:.1f} %; statistics tables skipped {skipped_stats}; garbled skipped {skipped_garbled}")
rows.sort(reverse=True)
print("worst chunks (cells without a profile / cells, chunk, page, caption):")
for r in rows[:25]:
    print("  ", r)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(" ".join(r[2] for r in rows if r[0] >= args.min_missing), encoding="utf-8")
print("cells without profile in the 25 worst chunks:", sum(r[0] for r in rows[:25]), "of all", tot - hit)
