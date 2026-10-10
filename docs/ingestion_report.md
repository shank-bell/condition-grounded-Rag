# Ingestion pipeline report (offline path A-D), frozen 2026-09-30

> **Update 11 Oct 2026.** The tables below describe the RAW extraction of 30 September (the `profiles` table, which is never rewritten). Since 11 Oct the
> system reads the profiles through a code-only repair overlay (`profile_repairs`, `src/cgrag/ingestion/repair.py`, `scripts/repair_store.py`,
> `[features] profile_repair`): 5,383 of the 10,275 profiles have at least one field corrected from their own table cell, caption, block heading or the paper's
> own definitions (setting 3,321 fields, task 2,119, data set 915, model 467, language 433, model size 64), the retrieval cards were rebuilt from the repaired
> fields, and the table reader reads a header row repeated in the middle of a wide table. Measured effect: `docs/evaluation_ai_annotated.md`, section 4.1b.

Scope: the offline path of the architecture - A PDF loader, B section chunker, C Condition Profile Extractor (LLM), D BGE-M3
embedder, and the three stores (ChromaDB vectors, BM25 index, SQLite Condition Profile store). Everything here was measured on
the 28 papers in `data/papers` with `scripts/ingest_metrics.py` (no LLM; `--json data/index/ingest_metrics.json`) and
`scripts/store_summary.py`. Extraction model: Gemma 4 12B through Ollama, 8 parallel slots.

## What is in the stores
| | |
|---|---|
| Papers | 28 (all indexed, all with profiles) |
| Chunks | 1,385 (292 are recognised tables kept whole) |
| Vectors / BM25 documents | 1,385 / 1,385, 1,024-dimensional BGE-M3 vectors |
| Condition Profiles | **10,275** (11,865 extracted, 1,560 removed by the garbled-table rule below, 30 by the junk-metric rule on 1 Oct) in 284 chunks |
| Retrieval cards (1 Oct) | 284 chunks carry a card built from their profiles (mean 245 characters), a second vector each; `scripts/reindex_cards.py` rebuilds them in 16 s without an LLM (run it after `reclean_profiles.py --apply`; `ingest.py` builds them automatically) |
| Integrity | 0 profiles pointing to a missing chunk, 0 broken Methods links, 0 papers without an extraction log |
| Extraction time | ~114 min in total, median 2.2 min per paper (max 15 min, the two 60-page model reports) |

## Quality (measured on the stored profiles)
| Measure | Value |
|---|---|
| Fill rate: model / metric / value | 100 % |
| Fill rate: task / dataset / setting | 96.0 % / 94.8 % / 98.5 % |
| Fill rate: language / model_size / dataset_version | 83.9 % / 41.4 % / 4.4 % (unrecorded is legitimate: most tables do not state them) |
| Methods-chunk link | 75.2 % |
| Value present in its chunk | 100 % (a profile whose number is not in the chunk is dropped) |
| Decimal table cells that became a profile | 80.7 % of 12,545 cells in 267 result tables (84.7 % before the garbled tables were removed on purpose; the removed cells now count as misses) |
| Row label = the profile's model | 86.4 % |
| Language column = the profile's language | 97.9 % (2,407 cells) |
| Hand check by the assistant, 84 random profiles (before the clean-up) | ~58 % fully right, ~70 % model + value right, ~23 % wrong, ~7 % not verifiable |

The hand check is NOT the team's field-level F1; that comes from labelling job A (2026-10-01, ~300 profiles). The hand-check
errors were dominated by garbled tables and cut model names, both of which are now filtered, so the true figure should be a
little higher, but that is not measured and is not claimed.

## Decisions that shaped the pipeline (all approved 2026-09-30 unless marked)
1. **Layout add-on for tables and captions** (`pymupdf4llm` + `pymupdf-layout`, same vendor as PyMuPDF): a table is found by the
   layout model, its caption is paired by best assignment per page (stacked tables no longer steal each other's captions).
2. **Whole-table chunks**: a recognised table is one chunk (caption + header + rows); over 5,000 characters it is cut between
   rows with caption and header repeated.
3. **Headers applied in code**: the extractor reads "row label: column = value; ..." lines, so the LLM never matches a number
   to a column. Row labels that were split across lines are joined (blank label inherits the row above).
4. **Statistics tables are not read** (dataset sizes, parameter counts, carbon): 25 tables skipped on purpose.
5. **Clean-up rules after the LLM**: bare-number model_size and "Avg" language are emptied; "Dev"/"Test" in the dataset moves to
   setting; a metric that repeats the dataset becomes "score"; a profile whose model is a piece of a row label ("L", "MA"),
   a size word, a number or the dataset is dropped; carbon/energy figures are dropped (788 profiles).
6. **Garbled tables are not read** (2026-09-30 evening, autonomy): a table where `garble_ratio >= 0.10` (cells holding three or
   more separate numbers, glued numbers such as "0.000.10", ". 83.83" fragments) is skipped and its profiles removed with
   `scripts/reclean_profiles.py --apply`: 15 chunks, 1,560 profiles (T5 1,421; ELECTRA/GPT-3/LLaMA the rest). A wrong number
   in the store is worse than a missing one, because Stage 7 would compare it with real ones.
7. **Methods context is prose only**: a results table filed under "methods" is never used as the Methods link (this bug had cut
   LLaMA recall from 96 % to 18 % before it was fixed).
8. **References are not indexed** (appendices are; a results table pushed between the last reference and an appendix is kept).

9. **Junk metric names are dropped** (1 Oct): a metric that is one run of 20+ letters or longer than 45 characters (a mis-read header such as
   `mniorpasasatsdtateuravg`, 30 profiles of one IndicXTREME table) is not a result (`profile_extractor._junk_metric`).
10. **Retrieval cards** (1 Oct, `ingestion/cards.py`): see `docs/oct1_fixes_and_metrics.md` section 4. The profile store is read, not changed.

## Known limitations (frozen, to be stated in the paper)
- **Column-wise tables** (11 of 350 layout tables flagged by `scripts/scan_stacked.py`; looked at one by one on 1 Oct): only **two
  are real stacked / side-by-side merges** (RAG p6, DistilBERT p3 Tables 2+3 - parameter counts and timings, plus RoBERTa p9 Tables
  6+7). The rest are not column-wise: TinyBERT p11 is an ordinary table with three-line headers (parsed correctly), GPT-3 p19 is two
  blocks with a repeated header row, Llama 2 p6 mixes a figure and a table, Llama 2 p61 is prose examples. The merged ones stay in
  the index as text chunks (retrieval finds them and the answer can read them) but yield no reliable profiles. Decision: no
  un-stacker. NOTE: the DistilBERT GLUE table itself (Table 1) was extracted fine (31 profiles); an earlier claim in this project
  that "DistilBERT vs BERT-base on GLUE has no profile because the layout model missed its tables" was wrong.
- **Tables missed by the layout model**: ~15 result tables (DistilBERT T2/T3, RoBERTa T7, T5 T11/T13-15, RAG T2, mT5 T10/T11
  has no caption, ...). T5 cell recall is 22 % because its 1,400 appendix cells were garbled and removed.
- **Rows labelled by a citation** ("Devlin et al. (2018)" meaning mBERT) are not resolved to a model name.
- **Group-label fragments** in Llama 2 / LLaMA rows ("7B" under a family label): the wrong name is filtered, the right one is
  not recovered (only 2 size-only labels remain in the store).
- **Wrong column when a value repeats** in a row and **swapped pair cells** ("GPS / Acc"): a few per cent of profiles.
- **dataset_version is recorded for 4.4 %** of profiles because most papers do not write a version; Stage 6/7 treat an
  unrecorded version as "not stated", never as "different".
- 16 profiles of percentage metrics carry a value outside 0-100 (kept, flagged by the metrics script).

## How to reproduce
```
.\.venv\Scripts\python scripts\ingest.py            # all PDFs in data/papers (needs Ollama; nothing else may hold the index)
.\.venv\Scripts\python scripts\reclean_profiles.py --apply   # re-apply the clean-up rules to stored profiles (no LLM)
.\.venv\Scripts\python scripts\ingest_metrics.py --json data\index\ingest_metrics.json
.\.venv\Scripts\python scripts\store_summary.py
```
Backups made before the 2026-09-30 polish and before the garbled-table removal: `D:\backup_cgrag\index_before_polish`,
`D:\backup_cgrag\profiles_before_reclean.sqlite`.
