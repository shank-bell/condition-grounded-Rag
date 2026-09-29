# Condition-Grounded Scientific RAG — Project Context

Read this first in any new session on this project. It captures decisions made in chat that
aren't visible from the code alone. The full architecture is in
`condition_grounded_rag_architecture.html` (git-ignored, kept local/private — read it directly
for the design; don't re-derive it from here).

## What this is
BE major project, Dept. of ISE, BMSCE. Team: Shashank BU, Aditya Venkatesh Dhanakshirur, Tarun K.
Guide: Dr. Rajeshwari K. A RAG pipeline over CS/AI research papers with two novel contributions:
- **Stage 6, Applicability Agent** — checks whether retrieved evidence actually covers the
  conditions (language, dataset, model size, etc.) implied by the question; re-retrieves or
  warns if not.
- **Stage 7, Condition-Aware Contradiction Resolver** — tells a genuine contradiction between
  papers apart from an "explained" difference caused by different experimental conditions
  (e.g. SQuAD v1.1 vs v2.0).

Both read/write a shared **Condition Profile** store (SQLite) keyed to chunks, built by a new
**Condition Profile Extractor** (stage C, offline, LLM-based).

## Hard rules (do not relax these without the user saying so)
1. **Integrate everything, no matter what.** All 10 pipeline stages must be wired into one
   working `run(question)` path, not built/tested as isolated pieces. If a component underperforms,
   swap in a fallback (see `docs/backup_models_huggingface.md`) — never leave a stub and call it done,
   and never quietly drop a stage.
2. **Never commit `condition_grounded_rag_architecture.html`.** It's in `.gitignore` deliberately —
   unpublished research. Check `git check-ignore -v condition_grounded_rag_architecture.html`
   before any `git add .` if this rule ever seems at risk.
3. **The user pushes to GitHub themselves.** Don't run `git push` (or add/commit) on their behalf —
   give them the commands and let them run them. (The repo
   `https://github.com/shank-bell/condition-grounded-Rag` is **public**; the user knows this and has
   accepted it for everything except the architecture doc.)
4. **Follow the architecture strictly.** Any departure from the architecture doc must be listed under
   "Departures from the architecture" below and be OK'd by the user.
5. **Get a basic path working before polishing.** Run one real question end to end before perfecting any single
   stage (this was learned the hard way on day 3: hours went into the extractor before the online path ever ran).
6. **Agent stages are LLM agents (user requirement, 2026-09-29).** Stage 2 (Orchestrator) and Stage 6 (Applicability)
   are agents driven by a local open-weights LLM (Hugging Face models via Ollama; **no external APIs**). Plain code
   only supplies guardrails (ablation switches; a "covered" verdict must be backed by a recorded/quoted value; model,
   dataset_version and model_size are never LLM-decided) and the fallback when the LLM output is unusable. Each agent
   has its own model setting under `[agents]` and an on/off switch under `[features]` for the ablation. Stages 7 and 9
   are also labelled agents in the doc; they already run on an open-weights HF model (DeBERTa NLI) with code decisions —
   ask the user before making their decisions LLM-driven. Fine-tuning a model for an agent was discussed and deferred.

## Timeline
13 days from 2026-09-27 (so ends ~2026-10-10): ~10 days coding, last 3 for the paper.
- Days 1–7: build all 10 stages, full pipeline integrated by day 7.
- Days 8–9: pilot evaluation + ablations (scaled down from the doc's full plan — see below).
- Day 10: code freeze, final numbers.
- Days 11–13: write the paper (draft earlier where possible; don't touch results after day 10).

**Scope trimmed from the doc for time:**
- Evaluation: ~30 papers / ~30 questions / ~50 labelled pairs (not the doc's 100/150/300).
- Stage 1 classifier: SciBERT 2-head trained on LLM-written questions (silver labels) instead of hand labels
  (no labelled training data exists); Gemma zero-shot is used until it is trained.

## Machines
- **College PC (in use since 2026-09-29, host DESKTOP-TKUPKK8):** RTX PRO 4000 Blackwell **24 GB**, Xeon w5-2545
  (12c/24t), 63.5 GB RAM, Windows 11. Project venv at `.venv` (Python 3.12.7, torch 2.11+cu128); Node 24 / npm 11.
- **Laptop:** RTX 4050 **6 GB**, Ryzen 7 7445HS, 15 GB RAM (was the only machine before). The architecture doc says
  "all on RTX 4050 6 GB" — that claim is only true if the final numbers are produced on the laptop. Decide which machine
  produces the paper's numbers before day 8.

### Ollama on the college PC — gotcha that cost an hour
Ollama 0.34.4's GPU discovery crashed (0xc0000005) and it silently ran on the **CPU** (~11 tok/s) because
`C:\Windows\System32\MSVCP140.dll` was a 2018 build (14.13); Ollama needs >= 14.44. Fixed by installing the current
Visual C++ Redistributable (`winget install Microsoft.VCRedist.2015+.x64`, now 14.51) and restarting Ollama.
Always check `ollama ps` shows **100% GPU** and `nvidia-smi` shows memory in use; CPU fallback looks like 10x slower.
The server is started by hand with more parallel slots (not persistent; after a reboot start it again):
`$env:OLLAMA_NUM_PARALLEL="8"; & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve`

### LLM choice
Pulled: `gemma4:e2b` (the doc's model), `gemma4:e4b`, `gemma4:12b`. `config.local.toml` (git-ignored, machine-specific)
sets `model = "gemma4:12b"`, `parallel = 8` and all devices to `cuda`. The LLM is one switch (`[llm].model`); the final
choice is made after the end-to-end pilot. Fine-tuning Gemma is NOT part of the architecture; only consider it after
everything else is done (needs gold data that is not the evaluation set; QLoRA fits 24 GB).
Measured on this GPU (single request, synthetic prompts): decode e2b 160-210 tok/s, e4b 105-125, 12b ~61; one
profile-extraction call 2.0 / 3.2 / 6-7 s; a 300-token answer ~1.6 / 2.7 / 5.7 s. Real pipeline on 12b with every
agent on: 4-15 s per question when warm (first question ~30 s while models load).

## State of the build (2026-09-29, day 3; everything below is uncommitted)
**Offline path** (`src/cgrag/ingestion`, `models.py`, `stores`): PDF loader (`pdf_loader.py`: rows merged, headings,
tables) -> section chunker (`chunker.py`, IMRaD tags, 2000/200) -> `tables.py` (aligns each number with its column
header) -> Condition Profile Extractor (`profile_extractor.py`, LLM + JSON schema) -> BGE-M3 embeddings (`models.py`)
-> ChromaDB + BM25 + SQLite profile store (`stores/`). `scripts/ingest.py` runs all of it.
**Online path** (`src/cgrag/pipeline`): stages 1-10 in one `run(question)` (`run.py`), both loops (6->4 re-retrieve,
9->8 regenerate); FastAPI (`api/main.py`: /query, /upload, /papers, /health); React + Vite UI in `frontend/`;
`scripts/ask.py` prints everything the pipeline decided.
**Tests:** 81 unit tests (`python -m pytest`), no GPU or LLM needed. Real-question smoke scripts are in the session
scratchpad only (basic RAG, and all agents on): re-create them from `scripts/ask.py` if needed.
**Agents:** Stage 2 (`pipeline/orchestrator.py`) is an LLM planner that decides refine / decompose / conflict check and
re-plans once after weak retrieval; Stage 6 (`pipeline/applicability.py`) looks up recorded profiles, has the LLM judge
what they cannot settle, writes its own follow-up search query and re-searches once. Both fall back to fixed rules /
the deterministic matcher when the LLM output is unusable.
**Proven on real papers:** basic RAG and all agents run end to end on real papers; answers were checked against the
papers (SQuAD / XNLI / BERT numbers, Kannada question correctly warned as not covered).
**Not done yet:** SciBERT classifier not trained (code + data generator ready: `scripts/make_query_training_data.py`,
`scripts/train_query_classifier.py`); no evaluation (gold sets, baselines, RAGAS not installed); extractor accuracy only
spot-checked on BERT's tables, no hand-labelled field-level F1; reranker threshold (-2.0) uncalibrated; UI never opened
in a browser; /upload untested; 10 of the 28 PDFs in `data/papers` are ingested (263 chunks, 1721 profiles); the other 18
are not (run `scripts/ingest.py` for them; it skips papers already indexed).

## Departures from the architecture (all need the user's OK; ones marked * were not yet approved)
- 12B Gemma instead of E2B (user's choice for the pilot; one-line switch).
- Stage 1 uses Gemma zero-shot until the SciBERT 2-head classifier is trained (approved).
- *Chunker leaves the References section out of the index.
- *Extractor also reads Discussion chunks and table-dense chunks in other sections (heading tags are heuristic) and
  stops at 30 chunks per paper.
- *Extractor extras around the LLM: rows of numbers are aligned with their column headers first; afterwards profiles
  whose number is not in the chunk, that have no system/metric, or that are dataset statistics are dropped.
- *Stage 1 also picks up dataset / version / model / language names written literally in the question.
- *Stage 7 reports only conflicts about the model/dataset the question names, once per pair of papers; text-only
  disagreement needs both sentences relevant, sharing words, and contradiction in both NLI directions.
- *Stage 9 also accepts a sentence whose numbers all occur in the cited source when NLI is not contradicting it
  (NLI models are weak on tables).

## How to run
```
# Ollama (see gotcha above), then:
.\.venv\Scripts\python scripts\ingest.py [paper stems] [--force]     # do NOT run while the API/pipeline is running
.\.venv\Scripts\python scripts\ask.py "Does BERT work well for Kannada question answering?"
.\.venv\Scripts\uvicorn cgrag.api.main:app --port 8000               # then: cd frontend; npm run dev
.\.venv\Scripts\python -m pytest
```
ChromaDB's persistent folder is single-process: never ingest and query at the same time.
Git push on this network sometimes fails once with "Recv failure: Connection was reset"; a plain retry works.

## Key files
- `config.toml` / `src/cgrag/config.py` — single source of truth for models, devices, paths, retrieval params, and the
  ablation flags in `[features]` (applicability, contradiction, critic, profile_in_context). Machine overrides go in
  git-ignored `config.local.toml`.
- `src/cgrag/schemas.py` — frozen shared Pydantic contracts every stage must use. Extend, don't fork.
- `src/cgrag/llm.py` — `OllamaLLM` wrapper (`.chat`, `.structured`).
- `scripts/bench_llm.py` — LLM latency benchmark. `docs/backup_models_huggingface.md` — verified fallback models.

## Biggest known risks, in order
1. **Profile Extractor quality** — the 12B model is unreliable at matching numbers to table columns, so rows are
   aligned in code first; still only spot-checked. Do the hand-check on ~10 papers (field-level F1) before trusting
   either contribution. Weak spots seen: multi-line/grouped headers, dataset-statistics tables, per-category
   breakdown tables, language/model_size rarely recorded. Concrete failures found on the XLM-R paper (1911.02116): the NER
   table (en/nl/es/de columns) is stored as dataset "XNLI"; MLQA Hindi numbers are stored under "GLUE" with EM called
   "accuracy"; the 15-language XNLI table is missing, so "mBERT accuracy on XNLI for Hindi" has no profile. Investigate
   wide multilingual tables and dataset attribution (caption/header context) next.
2. **Evaluation labelling** (~50 contradiction pairs, ~30 questions) is people-hours, not something code fixes.
3. **Stage 1 classifier** — silver labels only; check its accuracy on the team's real questions.
4. **Latency** is no longer a problem on the college PC (only if the laptop is used for the final runs).

## Next steps
1. Finish ingesting the 28 papers, then re-run `scripts/ask.py` on a spread of questions; hand-check profiles of ~10 papers.
2. Generate training questions and train the SciBERT classifier.
3. Build the evaluation sets and ablation runner; install RAGAS then (heavy dependencies; dry-run first).
4. Open the UI in a browser and test upload.
5. Decide the final LLM (12B vs e4b vs e2b) from pilot quality and latency; decide the machine for final numbers.
