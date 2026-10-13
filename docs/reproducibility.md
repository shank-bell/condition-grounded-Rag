# Reproducibility guide (11 October 2026, updated 12 October)

Everything in `docs/evaluation_ai_annotated.md` can be re-run from this repository, the 28 development papers and one public benchmark. This page lists
what is needed, in the order it is needed, and which results depend on files that are **not** in the repository (and why).

## 1. Machine and software

| Item | Value used for every reported number |
|---|---|
| GPU / CPU / RAM | NVIDIA RTX PRO 4000 Blackwell 24 GB, Xeon w5-2545 (12 cores), 63.5 GB, Windows 11 |
| Python | 3.12.7; the exact package versions are in `requirements-lock.txt` (`pip freeze`, 128 packages); torch 2.11 + CUDA 12.8 |
| LLM server | Ollama **0.34.4** for every number measured before the restart of the machine on 11 Oct 2026 (17:07), **0.35.0** from then on (tags `oct11i`, `oct11j`, `oct11k`) and **0.40.3** after the restart of 12 Oct (13:03; tag `oct12a` and later). The server updated itself each time; the Stage 1 reading of 3 of the 30 gold questions differs between 0.34.4 and 0.35.0 (evaluation document 4.7f). Started by hand with `OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=3 OLLAMA_KEEP_ALIVE=60m` for question answering and `OLLAMA_NUM_PARALLEL=8` for extraction (the tray application's own start uses the defaults) |
| LLMs (Ollama tags and ids) | `gemma4:12b` (4eb23ef187e2: answer, Stage 1, Stage 3, extractor, judge baseline), `gemma4:e2b` (7fbdbf8f5e45: Stage 2 and Stage 6 agents), `llama3.1:8b` (46e0c10c039e: independent judge of the faithfulness study only) |
| Encoders (Hugging Face) | `BAAI/bge-m3` (embedder), `BAAI/bge-reranker-base` (Stage 5), `cross-encoder/ms-marco-MiniLM-L-6-v2` (Stage 7 text path), `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (NLI, Stages 7 and 9), `allenai/scibert_scivocab_uncased` fine-tuned by `scripts/train_query_classifier.py` (Stage 1) |
| PDF reading | PyMuPDF with `pymupdf4llm` + `pymupdf-layout` (table recognition) |
| Decoding | temperature 0 for extraction, judges and baselines; 0.2 for the answer; Ollama batching makes GPU runs repeatable only up to small numeric noise |

The number of parallel extraction workers (`[llm] parallel`) must match the server's slots (`OLLAMA_NUM_PARALLEL=2`: `parallel = 3` in `config.local.toml`): with 8 workers on
2 slots the requests queued past the 240 s request timeout on dense tables (12 Oct evening).

`config.toml` holds every default; `config.local.toml` (git-ignored) holds the machine's overrides; `CGRAG_OVERLAY=<toml>` merges one more file last
(`config/bench_metalead.toml` sends all paths to the benchmark's own folders so the 28-paper index is never touched).

## 2. Data

| Data | Where | In the repository? |
|---|---|---|
| 28 development papers (arXiv ids = file names, see `data/index/...` and the table in `docs/ingestion_report.md`) | `data/papers/*.pdf` | no: third-party documents, download them by arXiv id |
| MetaLead benchmark: 43 papers + 3,568 human-annotated results (Timmer, Bölücü, Wan; EACL 2026) | `python scripts/download_metalead.py` -> `data/bench/metalead/` | no (script only; the annotation file is the authors' work, github.com/RoelTim/metalead) |
| Papers with Code data set names (optional, 4,735 names) | `python scripts/build_pwc_names.py` -> `data/bench/pwc/` | no (CC-BY-SA, built on demand; the repair works without it) |
| Papers with Code leaderboards, the independent answer key of section 4.9 (155,456 rows; the snapshot of 28 July 2025, CC-BY-SA-4.0, 21.8 MB) | downloaded by `scripts/eval_pwc_gold.py` from `huggingface.co/datasets/pwc-archive/files` -> `data/bench/pwc/evaluation-tables.json.gz` | no (evaluation only; nothing from it is ever written into a profile) |
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
python scripts/rescue_garbled.py --apply              # only for a store ingested before 11 Oct night (ingest.py now reads these tables itself): T5 Table 14 and ELECTRA Table 8 had been skipped as garbled; needs Ollama; adds 290 profiles (T5 203, ELECTRA 87)
python scripts/rescue_garbled.py --apply --empty      # 12 Oct: also every wanted chunk that has no profile (a first reading that failed; tables whose headers name no metric are kept as 'score'); +171 profiles
python scripts/cell_recall.py                          # (data/index/_logs/cell_recall.py, not in the repository) share of table cells with a profile; writes the chunks below 90 %
python scripts/rescue_garbled.py --apply --complete <chunk ids>   # adds ONLY the numbers that are missing, with new ids after the existing ones (the labelled Stage 7 pairs refer to profile ids; check that 600 of 600 references still hold)
python scripts/rescue_garbled.py --apply --replace <chunk ids> [--min-ratio 0.95]   # a chunk whose table the reader now reads differently: swap its profiles (never for a chunk a labelled pair refers to)
python scripts/rescue_garbled.py --apply --complete <chunks with a view of 31 results or more>   # 12 Oct evening: with the output cap at 4,500 tokens (it was 3,000: the last result of every full view was cut) ; data/index/_logs/chunks_big_views.txt is the list
python scripts/fix_continued_headers.py --apply        # 12 Oct evening: the continuation chunks of a long table get the header rows the chunker did not repeat (the API must be stopped; run before reindex_cards.py)
python scripts/resolve_datasets.py                    # a table that names no data set: read it from the paper's own text (needs Ollama; about 2 min; writes data/index/dataset_context.json)
python scripts/repair_store.py --apply                # the profile repair as an overlay (raw rows are untouched; reads dataset_context.json); --clear removes it
python scripts/reindex_cards.py                       # retrieval cards from the repaired profiles

# 2. the public benchmark (about 2 h)
python scripts/download_metalead.py
set CGRAG_OVERLAY=config/bench_metalead.toml
python scripts/ingest.py
python scripts/rescue_garbled.py --apply --empty    # 12 Oct: the same reading steps for the benchmark store (export its chunks first: sampling.export_chunks), then --complete for the chunks of low cell recall,
python scripts/resolve_datasets.py --force ; python scripts/repair_store.py --apply   #   data/index/_logs/bench_sweep.sh is the whole sequence
python scripts/eval_metalead.py                       # coverage / agreement of the pipeline (raw and repaired) against the human gold
python scripts/baseline_llm_extract.py --workers 8     # plain-LLM baseline, same model, MetaLead's own prompt (about 1 h with 8 parallel slots)
python scripts/eval_metalead.py --pred data/bench/metalead/baseline_gemma.json --name baseline

# 3. the independent leaderboard key and the fresh samples (12 Oct)
python scripts/eval_pwc_gold.py --split all           # Papers with Code leaderboards (downloads the archive of 28 July 2025 once; 22 of the 28 papers; strict reading; section 4.9)
python scripts/make_extraction_sample.py build --n 100 --seed 20261012 --dir extraction_test4   # E4; then label labels.jsonl and: ... score --dir extraction_test4
python scripts/pass_table.py oct11g oct11k oct12a oct12c oct12d oct12e oct12f   # the passes side by side (tags of eval/labels files)

# 4. every other table: docs/evaluation_ai_annotated.md, section 8
```

Seeds that matter: pair sets `make_heldout_pairs.py --seed 1016 / 1017 / 1018 / 1019 / 1020`; extraction test sets `make_extraction_testset.py --seed 1110` (E1, 120
profiles of the 28 papers) and `--seed 2011` (E2, 100 profiles of the benchmark papers; the hashes of the code at that moment are in
`data/labelling/private/extraction_test2/code_hashes.txt`).

## 4. What is deterministic and what is not

* Everything that is **code only** (chunking, table reading, the repair, grounding, Stage 7's decision, all scorers) is deterministic: the same inputs give
  the same outputs; unit tests: `python -m pytest` (334 tests, no GPU or LLM needed).
* Everything that **calls a language model** (extraction, Stage 1 reading, the agents, the answer, the judges, the plain-LLM baseline) is run at temperature 0 but
  is not bit-repeatable across GPUs or Ollama versions; the repeat runs of 8 October (section 5 of the evaluation document) differ in 0 to 1 decisions of 30, and a
  change of the server version from 0.34.4 to 0.35.0 changed the reading of conditions in 3 of 30 questions (4.7f); 0.35.0 to 0.40.3 is measured by the passes `oct12a` to `oct12e` (all on 0.40.3). Quote a result together with the server version.
* One online-path source of non-determinism was found and removed on 11 Oct: Stage 1 chose among data set names of one length by the order of a Python `set`, which
  follows the process's string hash (the same question came back as "SQUAD" or "SQuAD" in different runs). For the same reason set `PYTHONHASHSEED` when a result must be bit-identical.
* The answer keys are written by one AI labeller. Swapping in human labels is a re-run of the same scorers (section 8 of the evaluation document).

## 5. Licences and ethics

The repository holds code, configuration, aggregated results and documents only. Third-party papers, the MetaLead annotations and the Papers with Code list
are downloaded by the user under their own licences and are git-ignored (`data/bench/`, `data/papers/`, `data/labelling/`). No personal data is processed.
The labelling was done by an AI assistant; no human subjects were involved.
