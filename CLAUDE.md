# Condition-Grounded Scientific RAG — Project Context

Read this first in any new session on this project. It captures decisions made in chat that
aren't visible from the code alone. The full architecture is in
`condition_grounded_rag_architecture.html` (git-ignored, kept local/private — read it directly
for the design; don't re-derive it from here).

## What this is
BE major project, Dept. of ISE, BMSCE. Team: Shashank BU, Adarsh Kumar,Aditya Venkatesh Dhanakshirur, Tarun K.
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
   before any `git add .` or push; it must print a rule.
3. **The user pushes to GitHub themselves.** Default: give them the commands and let them run them. Commit/push only
   when the user explicitly asks in that turn (on 2026-09-29 they did once: 55 commits, one per file, pushed to
   `main`). Commit messages carry no tool-attribution or co-author lines. The repo
   `https://github.com/shank-bell/condition-grounded-Rag` is **public**; the user knows this and has accepted it for
   everything except the architecture doc. Author identity in history: `shank-bell <shashankbelludi1@gmail.com>`
   (this PC has no git identity configured; pass it per command with `git -c user.name=... -c user.email=...`).
4. **Follow the architecture strictly.** Any departure from the architecture doc must be listed under
   "Departures from the architecture" below and be OK'd by the user.
5. **Get a basic path working before polishing.** Run one real question end to end before perfecting any single
   stage (learned the hard way on day 3: hours went into the extractor before the online path ever ran).
6. **Agent stages are LLM agents (user requirement, 2026-09-29).** Stage 2 (Orchestrator) and Stage 6 (Applicability)
   are agents driven by a local open-weights LLM (Hugging Face models via Ollama; **no external APIs**). Plain code
   only supplies guardrails (ablation switches; a "covered" verdict must be backed by a recorded/quoted value; model,
   dataset_version and model_size are never LLM-decided) and the fallback when the LLM output is unusable. Each agent
   has its own model setting under `[agents]` and an on/off switch under `[features]` for the ablation. Stages 7 and 9
   are also labelled agents in the doc; they already run on an open-weights HF model (DeBERTa NLI) with code decisions —
   ask the user before making their decisions LLM-driven. Fine-tuning a model for an agent was discussed and deferred
   (low payoff, no labelled plans, unproven serving path for a tuned Gemma 4 in Ollama).

## Timeline
13 days from 2026-09-27 (so ends ~2026-10-10): ~10 days coding, last 3 for the paper. Day 3 = 2026-09-29.
- Days 1–7: build all 10 stages, full pipeline integrated by day 7 (done on day 3; quality work remains).
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
`$env:OLLAMA_NUM_PARALLEL="8"; $env:OLLAMA_KEEP_ALIVE="60m"; & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve`

### LLM choice
Pulled: `gemma4:e2b` (the doc's model), `gemma4:e4b`, `gemma4:12b`. `config.local.toml` (git-ignored, machine-specific)
sets `model = "gemma4:12b"`, `parallel = 8` and all devices to `cuda`; `config.toml` still defaults to e4b. The LLM is one
switch (`[llm].model`), plus one optional model per agent (`[agents]`); the final choice is made after the end-to-end
pilot. Fine-tuning Gemma is NOT part of the architecture; only consider it after everything else is done.
Measured on this GPU (single request, synthetic prompts): decode e2b 160-210 tok/s, e4b 105-125, 12b ~61; one
profile-extraction call 2.0 / 3.2 / 6-7 s; a 300-token answer ~1.6 / 2.7 / 5.7 s. Extraction of one paper on 12b with 8
parallel slots: ~1-2.5 min. Real pipeline on 12b with every agent on: 4-15 s per question when warm; the first question
after start is ~30 s while the embedder/reranker/NLI load.

## State of the build (2026-09-30, day 4; GitHub has the 56 commits of 2026-09-29, everything since is uncommitted)
**Offline path** (`src/cgrag/ingestion`, `models.py`, `stores`): PDF loader (`pdf_loader.py`: rows merged, headings,
tables) -> section chunker (`chunker.py`, IMRaD tags, 2000/200) -> `tables.py` (aligns each number with its column
header) -> Condition Profile Extractor (`profile_extractor.py`, LLM + JSON schema, per-table views, parallel calls,
grounding + non-result filters) -> BGE-M3 embeddings (`models.py`) -> ChromaDB + BM25 + SQLite profile store
(`stores/`). `scripts/ingest.py` runs all of it (BM25 is rebuilt at the end of a run).
**Online path** (`src/cgrag/pipeline`): stages 1-10 in one `run(question)` (`run.py`), both loops (6->4 re-retrieve,
9->8 regenerate) plus a re-plan after weak retrieval; FastAPI (`api/main.py`: /query, /upload, /papers, /health, /jobs);
React + Vite UI in `frontend/`; `scripts/ask.py` prints everything the pipeline decided.
- Stage 1 `query_understanding.py`: LLM zero-shot (intent, complexity, conditions) + conditions cleaned/validated + names
  found literally in the question (vocabulary of the profile store, language list); SciBERT 2-head (`intent_classifier.py`)
  auto-loads once trained.
- Stage 2 `orchestrator.py`: LLM planner agent (refine / decompose / conflict check), guardrails, fixed-rule fallback,
  re-plan once after weak retrieval. Stage 3 `refinement.py`: rewrite + up to 3 sub-questions, re-appends lost conditions.
- Stage 4 `retrieval.py`: dense + BM25, RRF k=60, top 20, soft section boost. Stage 5 `rerank.py`: MiniLM cross-encoder,
  threshold -2.0, keep 10, weak flag.
- Stage 6 `applicability.py`: profile lookup -> LLM judges what profiles cannot settle (grounded, strict fields) -> own
  follow-up search query -> re-search once -> scope warning; per-condition coverage (as in the doc).
- Stage 7 `contradiction.py` + `conditions.py`: numeric trigger (same model family/dataset/metric, >2% apart) + strict text-only
  NLI; classes GENUINE / EXPLAINED / NOT_COMPARABLE decided from recorded conditions (dataset_version, model_size,
  language, setting tags; task is not compared); only conflicts about the named model/dataset; one per paper pair.
- Stage 8 `generate.py`: top 5 chunks + passages that cover a requested condition + conflict chunks; recorded results
  listed per source, most relevant to the question's conditions first; scope warning and conflict notes in the prompt.
- Stage 9 `critic.py`: NLI per sentence vs the cited chunk (+ other sources: mis-citation repair), accepts a sentence
  backed by a recorded profile (number + metric + model, dev/test consistent) or whose numbers all occur in the source
  when NLI does not contradict; skips "the sources do not say" / conflict-echo sentences; regenerates once.
**Tests:** 105 unit tests (`python -m pytest`), no GPU or LLM needed. Scripts (all need Ollama; none while an ingest or the
API holds the index): `scripts/smoke_questions.py` (5 questions through the whole pipeline, per-stage timings, `--set
KEY=VALUE` to compare model sizes per use case), `scripts/ingest_metrics.py` (store integrity, fill rates, table-cell recall,
row-label accuracy), `scripts/audit_sample.py` (random profiles next to their source row, for a hand check),
`scripts/table_detail.py <paper>` / `scripts/debug_view.py <chunk_id>` (why a table lost results), `scripts/reclean_profiles.py`
(re-apply the extractor's clean-up rules to stored profiles, no LLM), `scripts/upload_test.py` (API end to end).
**Ingested (2026-09-30):** all 28 PDFs in `data/papers`: 1,385 chunks (292 recognised tables), 11,865 profiles, ~114 min of
12B extraction (median 2.2 min/paper, max 15 min); ChromaDB / BM25 / profile store consistent, 0 broken links, vectors 1024-d.
Profile fields filled: model / metric / value 100%, task 96%, dataset 95%, setting 98%, language 83%, model_size 37%,
dataset_version 3.8%, Methods link 66%. Table-cell recall 84.7% of 12,545 decimal cells in 267 result tables (25 statistics
tables skipped on purpose); row label = profile's model on 86%; language column = profile's language on 98% (2,407 cells).
Hand check of 84 random profiles by me (not the team): ~58% fully right, ~70% right on model + value, ~23% wrong, 7% not
verifiable. Errors: garbled tables (T5 / GPT-3 appendix), model names cut at column borders for group-labelled rows
(Llama 2, LLaMA: "L", "MA", "acon" - now filtered, but the real name is lost), wrong column when a value repeats, pair
cells whose order is swapped ("GPS / Acc"), junk in setting / model_size. A row labelled by a citation ("Devlin et al.
(2018)" = mBERT) is not recognised as mBERT. ~15 result tables are still missed by the layout model (DistilBERT T2/T3, RoBERTa
T7, T5 T11/T13-15, RAG T2, mT5 T10/T11, ...).
**Proven on real papers:** all stages run end to end on the 28 papers; warm questions 7-14 s (median 8.8 s) with the two
agents on gemma4:e2b and the answer on 12b, first question ~30-47 s; /upload of a new PDF worked (111 s for a 15-page paper),
the new paper was queryable at once (removed again afterwards). E2B agents give the same coverage decisions as 12B agents
(-2.4 s per question) but plan worse (E2B over-refines simple lookups, never decomposes a comparison). E2B on Stage 1 is
NOT good enough (put "mBERT" into model_size, "XNLI" into task -> false scope warnings), so Stage 1 stays on 12B.
**Not done yet:** SciBERT classifier not trained (`scripts/make_query_training_data.py`, `scripts/train_query_classifier.py`
ready); no evaluation (gold sets, baselines, RAGAS not installed); no team-labelled field-level F1; rerank threshold and
retrieval settings uncalibrated; UI never opened in a browser.

## Departures from the architecture (all need the user's OK; ones marked * were not yet approved)
- 12B Gemma instead of E2B (user's choice for the pilot; one-line switch). Stage 1 uses Gemma zero-shot until the SciBERT
  2-head classifier is trained (approved). Stage 2 and 6 are LLM agents with per-agent model settings (user requirement).
- *Chunker leaves the References section out of the index (appendices are kept; so is a results table that a float
  pushed between the last reference and an appendix heading).
- *(2026-09-30) Stage A finds tables and their captions with PyMuPDF's layout add-on (`pymupdf4llm` + `pymupdf-layout`,
  same vendor, AGPL); text and headings still come from PyMuPDF. A caption is paired with its nearest table (best
  assignment per page). An unnumbered line is a section heading only if the whole line is a section name (a bold
  run-in title like "Multilingual Masked Language Models" flipped a paper to "methods").
- *(2026-09-30) Stage B keeps each recognised table whole as a chunk of its own (caption + header + rows; over 5000 chars
  it is cut between rows with caption and header repeated) instead of cutting at 2000 characters.
- *(2026-09-30) Stage C reads a recognised table through its header, applied in code ("row label: column = value; ..."),
  so the LLM no longer matches numbers to columns; tables whose caption is about statistics / model sizes are not sent
  to the LLM; the "Methods" context given to the LLM is prose only (a results table filed under "methods" is never used);
  cap raised to 60 chunks per paper. Clean-up rules after the LLM (`_normalize` / `_is_result`, and
  `scripts/reclean_profiles.py` for stored profiles): a bare number as model_size and "Avg" as language are nulled;
  "Dev" / "Test" in the dataset field moves to setting; a metric that repeats the dataset name becomes "score"; a profile
  is dropped when its model is a piece of a row label ("L", "MA", "acon"), a size word, a number or the dataset itself, or
  when it reports carbon / energy figures (788 of 12,653 profiles were removed this way).
- *(2026-09-30) Stage 6 treats a translation direction ("English-to-German") as covered when a source records either of
  its two languages.
- Model size per use case (user rule 2026-09-30): the two agents run gemma4:e2b, the answer (Stage 8) gemma4:12b; the
  extractor stays on 12B (my choice, not measured against E2B yet). Stage 1 and Stage 3 have their own keys, still 12B.
- *Extractor also reads Discussion chunks and table-dense chunks in other sections (heading tags are heuristic) and
  stops at 60 chunks per paper (main-body result chunks first).
- *Extractor extras around the LLM: rows of numbers are aligned with their column headers first; afterwards profiles
  whose number is not in the chunk, that have no system/metric, or that are dataset statistics are dropped; dataset
  names carrying a version/split ("SQuAD 2.0 test") are split into dataset + version + setting.
- *Stage 1 also picks up dataset / version / model / language names written literally in the question.
- *The Orchestrator agent may add refinement for a question Stage 1 called simple (agent discretion); complex ones are
  always rewritten + decomposed; the applicability check cannot be skipped when conditions are named.
- *Stage 7 reports only conflicts about the model/dataset the question names, once per pair of papers; text-only
  disagreement needs both sentences relevant, sharing words, and contradiction in both NLI directions; free-text task
  is not compared; settings are compared by tags (dev/test/zero-shot/...).
- *Stage 8 adds passages found to cover a requested condition and shows the most relevant recorded results first.
- *Stage 9 also accepts profile-backed and number-grounded sentences, repairs mis-citations, skips conflict echoes.

## Open questions for the user (recommendation in brackets)
- A third "middle" complexity level? Doc and schema have two [keep two].
- Make Stages 7 and 9 decisions LLM-driven too? They already use an open-weights NLI model [leave as is].
- Joint condition coverage (a single finding matching all conditions) as an extra warning? Beyond the doc [propose, needs OK].
- Final LLM (12B / e4b / e2b) and per-agent models; which machine produces the paper's numbers.

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
- `config.toml` / `src/cgrag/config.py` — single source of truth for models, devices, paths, retrieval params, per-agent
  models (`[agents]`) and the ablation flags in `[features]` (orchestrator_agent, applicability_agent, applicability,
  contradiction, critic, profile_in_context). Machine overrides go in git-ignored `config.local.toml`.
- `src/cgrag/schemas.py` — frozen shared Pydantic contracts every stage must use. Extend, don't fork.
- `src/cgrag/llm.py` — `OllamaLLM` wrapper (`.chat`, `.structured`).
- `scripts/bench_llm.py` — LLM latency benchmark. `docs/backup_models_huggingface.md` — verified fallback models.

## Biggest known risks, in order
1. **Profile Extractor quality** — the 12B model is unreliable at matching numbers to table columns, so rows are
   aligned in code first; still only spot-checked. Do the hand-check on ~10 papers (field-level F1) before trusting
   either contribution. Weak spots seen: multi-line/grouped headers, dataset-statistics tables, per-category
   breakdown tables, language/model_size rarely recorded. Concrete failures found on the XLM-R paper (1911.02116): the NER
   table (en/nl/es/de columns) is stored as dataset "XNLI"; MLQA Hindi numbers are stored under "GLUE" with EM called
   "accuracy"; the 15-language XNLI table is missing, so "mBERT accuracy on XNLI for Hindi" has no profile; a suspicious
   "XLM-R accuracy 92.25" feeds a doubtful conflict. Investigate wide multilingual tables and dataset attribution
   (caption/header context) next.
2. **Evaluation labelling** (~50 contradiction pairs, ~30 questions) is people-hours, not something code fixes.
3. **Stage 1 classifier** — silver labels only; check its accuracy on the team's real questions.
4. Retrieval / rerank settings and the agents' decisions are unmeasured; the LLM sometimes rates a comparison question
   "simple" (the Orchestrator agent compensates).

## Next steps
1. (done 2026-09-30: all 28 papers in, wide multilingual tables fixed.) Left in the extractor: citation-labelled rows
   (map "Devlin et al. (2018)" to mBERT), group-label fragments, garbled appendix tables, ~15 missed result tables; and
   the team's hand-check of ~10 papers (field-level F1, needs their labels). Optional speed-up (needs the user's OK): fill
   table cells in code and let the LLM decide only the table-level conditions.
2. Generate training questions and train the SciBERT classifier (it should also fix "comparison rated simple", which lets
   the planner decompose comparisons - the E2B planner does not do that on its own).
3. Start the API and open the UI in a browser; test upload; consider warming the models at API start. UI decision
   (user, 2026-09-30): keep it a simple Claude-style chat page for now and polish it late; it must keep showing the
   coverage bar and conflict badges (stage 10 of the doc). The page already sends chat history for follow-ups.
4. Build the evaluation sets and an ablation runner (flags above, incl. agent vs rules); install RAGAS then (heavy
   dependencies; dry-run first); calibrate the rerank threshold.
5. Decide the final LLM / per-agent models and the machine for final numbers.
