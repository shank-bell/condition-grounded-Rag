# Condition-Grounded Scientific RAG

A question-answering system over research papers (computer science / AI) that treats the **experimental conditions** of a reported number - data set and
version, model and size, language, setting, split - as data of their own, so that it can say *when its evidence does not cover what was asked* and *why two
papers report different numbers for "the same" thing*. Everything runs locally (Ollama + Hugging Face models); no external API is part of the system.

| Component | What it does |
|---|---|
| **Condition Profile store** (offline) | every number a paper reports, with the conditions it was measured under (SQLite). Built by a local LLM from the paper's tables, then corrected with the tables' own captions, block headings and row / column labels by a code-only repair (`src/cgrag/ingestion/repair.py`, an overlay: raw rows are never rewritten). |
| **Stage 6, Applicability Agent** | before answering, checks whether the retrieved evidence really records every condition the question names (jointly: model *and* data set *and* language), re-retrieves, or warns and says which condition is missing. |
| **Stage 7, Condition-Aware Contradiction Resolver** | when two papers give different numbers, decides EXPLAINED (a recorded condition differs: data set version, split, model size, language, setting), GENUINE (same conditions, different numbers) or NOT COMPARABLE (conditions not recorded or not matching). Each profile is checked against its own table cell first (`pipeline/grounding.py`). |

The rest is a conventional RAG chain, wired end to end in one `Pipeline.run(question)`: Stage 1 question understanding (SciBERT intent / complexity +
LLM condition reader), 2 orchestrator agent, 3 refinement, 4 hybrid retrieval (BGE-M3 dense + BM25, RRF, retrieval cards), 5 cross-encoder rerank
(`bge-reranker-base`), 6 applicability, 7 contradiction, 8 grounded answer (Gemma 12B), 9 NLI claim critic, 10 response with a coverage bar and conflict
badges (React page in `frontend/`, FastAPI in `src/cgrag/api/`).

## Status and what the numbers mean

This is a research prototype (BE major project, BMSCE ISE). **No human labelled anything**: the answer keys of the evaluation were written by one AI
assistant from the PDFs, so every result is "against an AI-annotated key, one labeller, no kappa", and the labeller also helped build the system.
`docs/evaluation_ai_annotated.md` is the evaluation: method, corrections log (everything that went wrong), results, limits. Read its section 7 before
quoting any number. In short (development and clean numbers are marked there):

* Stage 6 scope warning: F1 92.7 on 30 questions (97.4 after error analysis); a simple LLM judge scores 91.4 - a tie; only the joint-coverage check matters.
* Stage 7: 220 of 250 labelled pairs right (88.0 %; "always EXPLAINED" 80.4 %, an LLM judge 64 %), false GENUINE calls 2 % against 16 % for the judge; it finds
  none of the 3 real contradictions in these sets. The accuracy gain over the trivial rule is small on clean sets (section 4.5f).
* Extraction: field-level F1 94.0 lenient / 90.0 strict after the repair on a clean sample (89.9 / 85.2 before, on another sample), but only 60.6 % / 39.4 % of
  the profiles are *completely* right. On the public MetaLead benchmark (human gold) the pipeline finds 88.4 % of the annotated results.
* Not above 85 %: completely right profiles, data set naming against human gold (73 %), Stage 7's naming of the differing condition (58 %), the gold
  evidence page in the retrieval stage's top 5 (78.6 %; it is among the answer's sources for 85.7 %, section 4.7c), real contradictions found (0 of 3).

## Quick start

```
python -m venv .venv && .venv\Scripts\pip install -r requirements-lock.txt     # Python 3.12
ollama pull gemma4:12b ; ollama pull gemma4:e2b                                # the LLMs the reported numbers used (about 9 GB of GPU memory with 2 slots); config.toml defaults to the smaller gemma4:e4b
python scripts/prewarm_files.py                                                # once after a reboot (slow cold file reads on some machines)
python scripts/ingest.py                                                       # PDFs in data/papers -> chunks -> profiles -> indexes (about 2 h for 28 papers)
python scripts/repair_store.py --apply ; python scripts/reindex_cards.py       # only for a store built before 11 Oct: ingest.py now repairs new papers itself
python scripts/ask.py "Does BERT work well for Kannada question answering?"    # prints every decision of every stage
uvicorn cgrag.api.main:app --port 8000                                         # API; then: cd frontend && npm install && npm run dev
python -m pytest                                                               # 258 unit tests, no GPU or LLM needed
```

`config.toml` holds every default and the ablation switches (`[features]`); machine overrides go in the git-ignored `config.local.toml`;
`CGRAG_OVERLAY=<toml>` merges one more file last. The 28 development papers are third-party documents and are **not** in the repository
(download them by arXiv id, see `docs/ingestion_report.md`); the same holds for the benchmark data (`scripts/download_metalead.py`).

## Layout

| Path | Content |
|---|---|
| `src/cgrag/ingestion/` | PDF loader, section chunker, table reader (`tables.py`), profile extractor, **repair**, retrieval cards |
| `src/cgrag/pipeline/` | Stages 1-10 (`run.py` wires them), `grounding.py` (a profile against its own table cell), `faithfulness.py` (optional rewrite loop, off) |
| `src/cgrag/knowledge/` | the two small lists the repair uses (benchmark -> task families, language codes) and an optional Papers-with-Code name list |
| `src/cgrag/stores/` | Chroma vectors, BM25, SQLite profile store with the repair overlay |
| `scripts/` | ingest, evaluation, label scoring, benchmark download, baselines (`docs/reproducibility.md` gives the order) |
| `eval/labels/` | aggregated results of every table of the evaluation (the answer keys themselves are private, on purpose) |
| `docs/` | evaluation (`evaluation_ai_annotated.md`), reproducibility, ingestion report, Stage 1 training report |
| `tests/` | unit tests |

## Reproducing

`docs/reproducibility.md`: machine and software, data, order of work, what is deterministic, licences. Seeds and scripts for every set of pairs and
samples are listed there and in section 8 of the evaluation document. Real human label files can be dropped in and scored with the same scripts.

## Licence

Not yet chosen by the authors; until then all rights are reserved by them. The papers and benchmark data used for evaluation are the property of their
authors and are not distributed here.
