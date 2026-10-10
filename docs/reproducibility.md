# Reproducibility guide (11 October 2026)

Everything in `docs/evaluation_ai_annotated.md` can be re-run from this repository, the 28 development papers and one public benchmark. This page lists
what is needed, in the order it is needed, and which results depend on files that are **not** in the repository (and why).

## 1. Machine and software

| Item | Value used for every reported number |
|---|---|
| GPU / CPU / RAM | NVIDIA RTX PRO 4000 Blackwell 24 GB, Xeon w5-2545 (12 cores), 63.5 GB, Windows 11 |
| Python | 3.12.7; the exact package versions are in `requirements-lock.txt` (`pip freeze`, 128 packages); torch 2.11 + CUDA 12.8 |
| LLM server | Ollama 0.34.4, started with `OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=3` for question answering and `OLLAMA_NUM_PARALLEL=8` for extraction |
| LLMs (Ollama tags and ids) | `gemma4:12b` (4eb23ef187e2: answer, Stage 1, Stage 3, extractor, judge baseline), `gemma4:e2b` (7fbdbf8f5e45: Stage 2 and Stage 6 agents), `llama3.1:8b` (46e0c10c039e: independent judge of the faithfulness study only) |
| Encoders (Hugging Face) | `BAAI/bge-m3` (embedder), `BAAI/bge-reranker-base` (Stage 5), `cross-encoder/ms-marco-MiniLM-L-6-v2` (Stage 7 text path), `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (NLI, Stages 7 and 9), `allenai/scibert_scivocab_uncased` fine-tuned by `scripts/train_query_classifier.py` (Stage 1) |
| PDF reading | PyMuPDF with `pymupdf4llm` + `pymupdf-layout` (table recognition) |
| Decoding | temperature 0 for extraction, judges and baselines; 0.2 for the answer; Ollama batching makes GPU runs repeatable only up to small numeric noise |

`config.toml` holds every default; `config.local.toml` (git-ignored) holds the machine's overrides; `CGRAG_OVERLAY=<toml>` merges one more file last
(`config/bench_metalead.toml` sends all paths to the benchmark's own folders so the 28-paper index is never touched).

## 2. Data

| Data | Where | In the repository? |
|---|---|---|
| 28 development papers (arXiv ids = file names, see `data/index/...` and the table in `docs/ingestion_report.md`) | `data/papers/*.pdf` | no: third-party documents, download them by arXiv id |
| MetaLead benchmark: 43 papers + 3,568 human-annotated results (Timmer, Bölücü, Wan; EACL 2026) | `python scripts/download_metalead.py` -> `data/bench/metalead/` | no (script only; the annotation file is the authors' work, github.com/RoelTim/metalead) |
| Papers with Code data set names (optional, 4,735 names) | `python scripts/build_pwc_names.py` -> `data/bench/pwc/` | no (CC-BY-SA, built on demand; the repair works without it) |
| AI-annotated answer keys (jobs A, B, C and the extraction test sets) | `data/labelling/` | **no, on purpose**: the keys and the system's verdicts are kept apart to keep the labels blind; only aggregated results are in `eval/labels/` |
| Aggregated results of every table of the evaluation document | `eval/labels/*.json` | yes |

## 3. Order of work

```
# 0. environment
python -m venv .venv && .venv\Scripts\pip install -r requirements-lock.txt
ollama pull gemma4:12b ; ollama pull gemma4:e2b ; ollama pull llama3.1:8b          # the last one only for scripts/eval_ragas.py
python scripts/prewarm_files.py                                                       # once after a reboot (cold file reads are slow on the college PC)

# 1. the 28-paper corpus (about 2 h on the machine above)
python scripts/ingest.py                              # PDF -> chunks -> profiles -> BGE-M3 + Chroma + BM25 + SQLite
python scripts/repair_store.py --apply                # the profile repair as an overlay (raw rows are untouched); --clear removes it
python scripts/reindex_cards.py                       # retrieval cards from the repaired profiles

# 2. the public benchmark (about 2 h)
python scripts/download_metalead.py
set CGRAG_OVERLAY=config/bench_metalead.toml
python scripts/ingest.py
python scripts/eval_metalead.py                       # coverage / agreement of the pipeline (raw and repaired) against the human gold
python scripts/baseline_llm_extract.py --workers 8     # plain-LLM baseline, same model, MetaLead's own prompt (about 1 h with 8 parallel slots)
python scripts/eval_metalead.py --pred data/bench/metalead/baseline_gemma.json --name baseline

# 3. every other table: docs/evaluation_ai_annotated.md, section 8
```

Seeds that matter: pair sets `make_heldout_pairs.py --seed 1016 / 1017 / 1018 / 1019`; extraction test sets `make_extraction_testset.py --seed 1110` (E1, 120
profiles of the 28 papers) and `--seed 2011` (E2, 100 profiles of the benchmark papers; the hashes of the code at that moment are in
`data/labelling/private/extraction_test2/code_hashes.txt`).

## 4. What is deterministic and what is not

* Everything that is **code only** (chunking, table reading, the repair, grounding, Stage 7's decision, all scorers) is deterministic: the same inputs give
  the same outputs; unit tests: `python -m pytest` (258 tests, no GPU or LLM needed).
* Everything that **calls a language model** (extraction, Stage 1 reading, the agents, the answer, the judges, the plain-LLM baseline) is run at temperature 0 but
  is not bit-repeatable across GPUs or Ollama versions; the repeat runs of 8 October (section 5 of the evaluation document) differ in 0 to 1 decisions of 30.
* The answer keys are written by one AI labeller. Swapping in human labels is a re-run of the same scorers (section 8 of the evaluation document).

## 5. Licences and ethics

The repository holds code, configuration, aggregated results and documents only. Third-party papers, the MetaLead annotations and the Papers with Code list
are downloaded by the user under their own licences and are git-ignored (`data/bench/`, `data/papers/`, `data/labelling/`). No personal data is processed.
The labelling was done by an AI assistant (Claude); no human subjects were involved.
