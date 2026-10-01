# Condition-Grounded Scientific RAG — Project Context

Read this first in any new session on this project. It captures decisions made in chat that
aren't visible from the code alone. The full architecture is in
`condition_grounded_rag_architecture.html` (git-ignored, kept local/private — read it directly
for the design; don't re-derive it from here).

## RESUME HERE (rewritten 2026-10-01 ~15:40; read this first)
Day 5 (1 Oct). Standing orders from the user (do NOT stop to ask them questions; decide, flag, report): fix the five "seen but not
investigated" items, provide the metrics, build the Excel sheets for the labelling; "you take the pivotal decisions". **All done; the
numbers and the reasoning are in `docs/oct1_fixes_and_metrics.md` (read it).** Labels are an ANSWER KEY, never training data.
- **Git:** the user runs the commits themselves in PowerShell, ONE FILE = ONE COMMIT, no attribution lines, identity per command
  (`$id = '-c','user.name=shank-bell','-c','user.email=shashankbelludi1@gmail.com'`, then `git @id commit -m msg -- file`). Give the
  block BEFORE starting a session and again at the end (`git status --short` for the exact list). On GitHub: everything up to
  c5820cf (the 25 files of 30 Sep). **Everything of 1 Oct is uncommitted.** The architecture doc is git-ignored: never commit it.
- **Machine gotchas found today:** (1) after a reboot / long idle run `python scripts/prewarm_files.py` first (first read of every file costs
  ~0.25 s on this PC: a cold `import transformers` took ~25 min; prewarmed: 6 s); (2) Ollama can wedge: kill python jobs, kill `ollama.exe`
  AND the orphan `llama-server.exe`, restart `OLLAMA_NUM_PARALLEL=8 OLLAMA_KEEP_ALIVE=60m OLLAMA_MAX_LOADED_MODELS=2 ollama serve`; every
  LLM call now times out after 240 s; (3) never run two processes that write the Chroma index; (4) `$env:TEMP` scripts: prefer the Write tool
  over heredocs with backslashes (they get mangled).
- **Departures from the architecture** (all behind flags): the five of 30 Sep are approved; the older eight and everything added since
  (escalation, profile-guided retrieval, retrieval cards, joint coverage, weak-flag withdrawal, Stage 7 language filter, Stage 9 changes,
  garbled-table skip, Stage 1 training details) are listed under "Departures" and wait for the user's explicit OK (delegated to my judgement).

What stands (details in the docs named):
- **Stage 1 SciBERT** trained and installed 30 Sep (`docs/stage1_training_report.md`): dev C (48 held out) 1.000 / 1.000, with typos 0.983 / 1.000.
- **Ingestion frozen** (`docs/ingestion_report.md`): 28 papers, 1,385 chunks, **10,275 profiles**, cell recall 80.7 %, row label = model 86.4 %.
- **Critic / Stage 7 / retrieval / Stage 6 fixed 1 Oct** (`docs/oct1_fixes_and_metrics.md`): critic false rejections (Kannada answer 3/5 -> 6/6
  claims supported); false text-only conflicts (need a decimal / percentage in both sentences; language filter); **retrieval cards**
  (`ingestion/cards.py`, `[retrieval] use_cards`): hit@5 69 -> 90 %, language + task questions 22.5 -> 60 %; **joint coverage**
  (`[features] joint_coverage`, `applicability._joint`): correct scope decisions 22 -> 32 of 32 (warning recall 5/12 -> 12/12, 0 false
  warnings); benchmark suites (GLUE = CoLA, MRPC ...) via `conditions.covers`; weak-evidence flag withdrawn when every named condition
  is recorded; sentence splitter keeps "vs." / "et al."; junk-metric rule (30 profiles).
- **Escalation measured: no accuracy gain, +0.4 s per question on average (+3.5 s where it fires, 4 of 32).** Flag still ON (the user's design).
  Decide after job B: `python scripts/eval_questions.py eval/labels/questions_gold.jsonl --set features.escalation=false --tag no_escalation`.
- **Start-up:** `Pipeline.warm_up()` (torch models sequentially, Ollama in parallel) runs in a background thread at API start
  (`/health` shows `warm.status`); first question 38.8 -> 10.0 s; API first query 6.2 s.
- **Column-wise tables:** only 2 of the "11" are real merges; no un-stacker (ingestion report). My 30 Sep claim that DistilBERT's GLUE table was
  missed was WRONG (it was extracted; the "profile missing" note was a false NLI conflict, now fixed).
- **Labelling kit built and tested** (`src/cgrag/labelling/`, `scripts/make_label_sheets.py`, `score_labels.py`, `eval_questions.py`,
  `docs/labelling_guide.md`): 8 workbooks in `data/labelling/` (git-ignored), private keys in `data/labelling/private/`. NOT yet opened in Excel
  (no LibreOffice here). Tomorrow/now: send the workbooks, collect `_DONE` files by 3 Oct, score with `score_labels.py`.

Open items / next:
1. The user's OK on the departures; the escalation decision (above).
2. Send the sheets; help the team label; then `score_labels.py A/B/C`, `eval_questions.py`, a second ablation on the team's questions.
3. UI in a browser (never opened), evaluation sets + ablation runner for the paper, RAGAS, final LLM / machine decision (see "Next steps").
4. Known weak spots to remember: the joint check trusts the profile store (a missed table -> a possible false warning; job B's "covered"
   questions will measure it); the store has some wrong profiles (e.g. BERT-large 89.1 F1 on SQuAD 2.0 in the ALBERT paper); citation-labelled
   rows ("Devlin et al." = mBERT) are not resolved.

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
7. **Autonomy (2026-09-30 evening).** The user delegated the evening's work ("you manage all things here, don't come asking
   me"). Decide, document, keep every departure flagged in this file and behind a config flag, and report at the end.
   Never stop mid-run to ask; the only outputs they expect are short status lines and the final report + git block.

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
`$env:OLLAMA_NUM_PARALLEL="8"; $env:OLLAMA_KEEP_ALIVE="60m"; $env:OLLAMA_MAX_LOADED_MODELS="2"; & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" serve`
(`MAX_LOADED_MODELS=2` keeps 12B and E2B resident together; the server that runs now was started with these values.)

### LLM choice
Pulled: `gemma4:e2b` (the doc's model), `gemma4:e4b`, `gemma4:12b`. `config.local.toml` (git-ignored, machine-specific)
sets `model = "gemma4:12b"`, `parallel = 8` and all devices to `cuda`; `config.toml` still defaults to e4b. The LLM is one
switch (`[llm].model`), plus one optional model per agent (`[agents]`); the final choice is made after the end-to-end
pilot. Fine-tuning Gemma is NOT part of the architecture; only consider it after everything else is done.
Measured on this GPU (single request, synthetic prompts): decode e2b 160-210 tok/s, e4b 105-125, 12b ~61; one
profile-extraction call 2.0 / 3.2 / 6-7 s; a 300-token answer ~1.6 / 2.7 / 5.7 s. Extraction of one paper on 12b with 8
parallel slots: ~1-2.5 min. Real pipeline on 12b with every agent on: 4-15 s per question when warm; the first question
after start is ~30 s while the embedder/reranker/NLI load.

## State of the build (2026-10-01 afternoon, day 5; GitHub = c5820cf; everything of 1 Oct is uncommitted)
**Offline path** (`src/cgrag/ingestion`, `models.py`, `stores`): PDF loader (`pdf_loader.py`: rows merged, headings,
tables) -> section chunker (`chunker.py`, IMRaD tags, 2000/200) -> `tables.py` (aligns each number with its column
header) -> Condition Profile Extractor (`profile_extractor.py`, LLM + JSON schema, per-table views, parallel calls,
grounding + non-result filters) -> retrieval cards (`cards.py`: a summary of what each chunk records, from its profiles) ->
BGE-M3 embeddings of chunk + card (`models.py`) -> ChromaDB (collections `chunks` and `cards`) + BM25 (text + card) + SQLite profile store
(`stores/`). `scripts/ingest.py` runs all of it (BM25 is rebuilt at the end of a run); `scripts/reindex_cards.py` rebuilds the cards of an
existing index from the profile store (16 s, no LLM; run it after `reclean_profiles.py --apply`).
**Online path** (`src/cgrag/pipeline`): stages 1-10 in one `run(question)` (`run.py`), both loops (6->4 re-retrieve,
9->8 regenerate) plus a re-plan after weak retrieval; FastAPI (`api/main.py`: /query, /upload, /papers, /health, /jobs);
React + Vite UI in `frontend/`; `scripts/ask.py` prints everything the pipeline decided.
- Stage 1 `query_understanding.py`: SciBERT 2-head (`intent_classifier.py`, trained 2026-09-30, installed at
  `data/index/query_classifier/`) gives intent + complexity; the LLM reads the conditions (cleaned/validated; a task word such as
  NLI / QA / NER the LLM put in the dataset field is moved to task) + names found literally in the question (vocabulary of the
  profile store, language list). Without a checkpoint the LLM produces all three zero-shot.
- Stage 2 `orchestrator.py`: LLM planner agent (refine / decompose / conflict check), guardrails, fixed-rule fallback,
  re-plan once after weak retrieval. Stage 3 `refinement.py`: rewrite + up to 3 sub-questions, re-appends lost conditions.
- Stage 4 `retrieval.py`: dense + BM25 (+ the card vectors for result / comparison / factual questions), RRF k=60, top 20, soft section
  boost. Stage 5 `rerank.py`: MiniLM cross-encoder reading card + chunk text, threshold -2.0 (calibrated 1 Oct: best trade-off), keep 10,
  weak flag (withdrawn by `run.py` when Stage 6 covers every named condition by recorded values).
- Stage 6 `applicability.py`: profile lookup -> LLM judges what profiles cannot settle (grounded, strict fields) -> own
  follow-up search query -> re-search once (with `profile_guided_retrieval` the re-search also adds chunks whose ONE profile records
  the missing condition together with all other requested conditions) -> with `escalation` the bigger model gives a second
  opinion on any remaining "not covered" -> JOINT COVERAGE guardrail (`_joint`, `[features] joint_coverage`): when two or three of model /
  dataset / language are named, one profile must record them together (benchmark suites: GLUE = CoLA, MRPC ...), else the most specific
  condition (language, dataset, model) is reported not covered and the warning says what IS recorded -> scope warning; per-condition
  coverage (as in the doc).
- Stage 7 `contradiction.py` + `conditions.py`: numeric trigger (same model family/dataset/metric, >2% apart) + strict text-only
  NLI; classes GENUINE / EXPLAINED / NOT_COMPARABLE decided from recorded conditions (dataset_version, model_size,
  language, setting tags; task is not compared); only conflicts about the named model/dataset/language; one per paper pair; a
  text-only conflict needs a decimal or percentage in both sentences (a year in a citation is not a result).
- Stage 8 `generate.py`: top 5 chunks + passages that cover a requested condition + conflict chunks; recorded results
  listed per source, most relevant to the question's conditions first; scope warning and conflict notes in the prompt.
- Stage 9 `critic.py`: NLI per sentence vs the cited chunk (+ other sources: mis-citation repair), accepts a sentence
  backed by a recorded profile (number + metric + model, dev/test consistent) or whose numbers all occur in the source
  when NLI does not contradict (a claim that names no metric is judged on number + model + setting); skips "the sources do not say" /
  conflict-echo sentences, list lead-ins ending in ":" and bare "CoLA: 56.3" lines; checks short "MuRIL: 67.8" lines; regenerates once.
**Tests:** 165 unit tests (`python -m pytest`), no GPU or LLM needed. Scripts (all need Ollama; none while an ingest or the
API holds the index): `scripts/smoke_questions.py` (5 questions through the whole pipeline, per-stage timings, `--set
KEY=VALUE` to compare model sizes per use case), `scripts/ingest_metrics.py` (store integrity, fill rates, table-cell recall,
row-label accuracy), `scripts/audit_sample.py` (random profiles next to their source row, for a hand check),
`scripts/table_detail.py <paper>` / `scripts/debug_view.py <chunk_id>` (why a table lost results), `scripts/reclean_profiles.py`
(re-apply the extractor's clean-up rules to stored profiles, no LLM), `scripts/upload_test.py` (API end to end).
Added the evening of 2026-09-30 (uncommitted until the user commits): Stage 1 campaign `scripts/build_stage1_dev.py`,
`make_template_questions.py`, `prepare_query_data.py`, `train_query_classifier.py` (rewritten), `install_query_classifier.py`;
table diagnostics `garble_report.py`, `table_finder.py <paper> <N>`, `raw_tables.py <paper> <page>`, `scan_stacked.py`; later that
evening `eval_stage1.py`, `store_summary.py`. Added 2026-10-01: measurement `eval_retrieval.py` (190 store-derived questions, hit@k / MRR /
threshold sweep, no LLM), `ablate_stage6.py` (32 scope questions x configurations), `measure_first_question.py`, `prewarm_files.py`,
`debug_critic.py`, `debug_retrieval.py`, `reindex_cards.py`; labelling `make_label_sheets.py`, `score_labels.py`, `eval_questions.py`
(see `docs/labelling_guide.md`); library `src/cgrag/labelling/` (xl.py, sampling.py, sheets.py, score.py).
**Ingested (2026-09-30, frozen, `docs/ingestion_report.md`):** all 28 PDFs in `data/papers`: 1,385 chunks (292 recognised tables),
**10,275 profiles** (11,865 extracted, 1,560 garbled-table profiles and 30 junk-metric profiles removed), ~114 min of 12B extraction (median 2.2 min/paper, max
15 min); ChromaDB / BM25 / profile store consistent, 0 broken links, vectors 1024-d. Profile fields filled: model / metric / value
100%, task 96%, dataset 94.8%, setting 98.5%, language 83.9%, model_size 41.4%, dataset_version 4.4%, Methods link 75.2%.
Table-cell recall 80.7% of 12,545 decimal cells in 267 result tables (25 statistics tables skipped on purpose; 84.7% before the
garbled ones were removed); row label = profile's model on 86.4%; language column = profile's language on 97.9% (2,407 cells).
Hand check of 84 random profiles by me (not the team): ~58% fully right, ~70% right on model + value, ~23% wrong, 7% not
verifiable. Errors: garbled tables (T5 / GPT-3 appendix), model names cut at column borders for group-labelled rows
(Llama 2, LLaMA: "L", "MA", "acon" - now filtered, but the real name is lost), wrong column when a value repeats, pair
cells whose order is swapped ("GPS / Acc"), junk in setting / model_size. A row labelled by a citation ("Devlin et al.
(2018)" = mBERT) is not recognised as mBERT. ~15 result tables are still missed by the layout model (DistilBERT T2/T3, RoBERTa
T7, T5 T11/T13-15, RAG T2, mT5 T10/T11, ...).
**Proven on real papers:** all stages run end to end on the 28 papers; warm questions 7-14 s (median 8.3 s on 2026-09-30 evening
with SciBERT, escalation and profile-guided retrieval on) with the two agents on gemma4:e2b and the answer on 12b, first question
~30-47 s; the Kannada NLI question is answered from IndicXNLI (mBERT 58.6, XLM-R 71.5, MuRIL 74.0, IndicBERT+Samanantar 74.7, coverage 100 %,
6/6 claims supported) while "XLM-R on XNLI for Kannada" gets the joint-coverage scope warning ("no source reports language = Kannada
together with model = XLM-R, dataset = XNLI"); 1 Oct: warm median 10.9 s over the five key questions, first question 10.0 s after warm-up
(38.8 s without), API first query 6.2 s; /upload of a new PDF worked (111 s for a 15-page paper),
the new paper was queryable at once (removed again afterwards). E2B agents give the same coverage decisions as 12B agents
(-2.4 s per question) but plan worse (E2B over-refines simple lookups, never decomposes a comparison). E2B on Stage 1 is
NOT good enough (put "mBERT" into model_size, "XNLI" into task -> false scope warnings), so Stage 1 stays on 12B.
**Not done yet:** no evaluation (gold sets, baselines, RAGAS not installed); no team-labelled field-level F1 and no human-question
test of the SciBERT (labelling 2026-10-01); rerank threshold and retrieval settings uncalibrated; UI never opened in a browser; the
escalation's latency cost and accuracy gain are not measured in an ablation yet (flags exist).

## Departures from the architecture (all need the user's OK; ones marked * were not yet approved)
(2026-09-30: the user delegated the decision on the five departures of that day - layout add-on, whole-table chunks, headers
applied in code, statistics-table skip, clean-up rules - with "do what you think is best"; they are approved. The eight
older ones marked * further down, and the flagged additions at the end of this list, still wait for an explicit OK.)
- 12B Gemma instead of E2B (user's choice for the pilot; one-line switch). Stage 1 uses Gemma zero-shot until the SciBERT
  2-head classifier is trained (approved). Stage 2 and 6 are LLM agents with per-agent model settings (user requirement).
- *Chunker leaves the References section out of the index (appendices are kept; so is a results table that a float
  pushed between the last reference and an appendix heading).
- (approved 2026-09-30) Stage A finds tables and their captions with PyMuPDF's layout add-on (`pymupdf4llm` + `pymupdf-layout`,
  same vendor, AGPL); text and headings still come from PyMuPDF. A caption is paired with its nearest table (best
  assignment per page). An unnumbered line is a section heading only if the whole line is a section name (a bold
  run-in title like "Multilingual Masked Language Models" flipped a paper to "methods").
- (approved 2026-09-30) Stage B keeps each recognised table whole as a chunk of its own (caption + header + rows; over 5000 chars
  it is cut between rows with caption and header repeated) instead of cutting at 2000 characters.
- (approved 2026-09-30) Stage C reads a recognised table through its header, applied in code ("row label: column = value; ..."),
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
- *(2026-09-30 evening, new) Stage C does not read garbled tables: a table with `tables.garble_ratio` >= 0.10 (cells that hold
  three or more separate numbers, glued numbers such as "0.000.10", ". 83.83" fragments, "53.84 9" split numbers) is skipped
  and its stored profiles are removed (`scripts/reclean_profiles.py`); a table of contents is not a table (dot leaders).
  Affects 15 chunks / ~1,560 profiles (T5 Table 16, GPT-3 appendix, LLaMA MMLU detail, ELECTRA Table 8, ...).
- *(2026-09-30 evening, new) Stage 1 training data: LLM-written questions + template questions with labels correct by
  construction + a blind LLM relabel that keeps only questions whose label the second pass confirms + an assistant-written
  dev set (`eval/stage1_dev.jsonl`, labels are the assistant's judgement, not gold). Training is automated and monitored by
  sub-agents (see `docs/stage1_campaign.md`). The silver-label SciBERT itself was approved on 2026-09-29.
- *(built 2026-09-30 evening; flags on in config.toml; tests in tests/test_escalation_and_profile_guided.py; need the user's OK)
  `[features] escalation` + `[agents] fallback_model` (empty = [llm].model): an agent on a small model hands a bad OUTCOME to the
  bigger one - the Orchestrator's unusable plan, a weak / thin retrieval (the bigger model re-plans), and any remaining "not covered"
  verdict of the Applicability agent (second opinion before a scope warning; the guardrails still apply). Never triggered by mere
  disagreement with the rules. `[features] profile_guided_retrieval`: Stage 6's re-search also asks the profile store which chunks
  have ONE profile that records the missing condition together with all the other requested conditions (so "XLM-R on XNLI for
  Kannada" still warns) and adds the best two after the reranker scored them. Plus two small matcher fixes: task abbreviations
  (NLI / QA / NER / MT ... = their long forms; equal short words match) in `conditions.values_match`, and a task word the LLM put in
  the dataset field is moved to task in Stage 1. Also `[llm] request_timeout` = 240 s (a wedged Ollama runner raised no error
  before and blocked jobs for 15 minutes).
- *(2026-09-30 evening, new) Stage 1 classifier training details: the validation score used for the composite is the LLM-written
  part (templates inflate it); training data include perturbed copies of the questions (typos, lower case, chatty prefixes); the
  held-out split C of the dev set is scored only at the end. Result and limits: `docs/stage1_training_report.md`.
- *(2026-09-30 evening, new) Column-wise / side-by-side tables are not un-stacked (looked at on 1 Oct: only 2 of the "11" are real merges);
  they stay as text chunks without reliable profiles (limitation, `docs/ingestion_report.md`).
- *(2026-10-01, new; each behind a flag or a documented rule; numbers in `docs/oct1_fixes_and_metrics.md`)
  **Retrieval cards** (`[retrieval] use_cards`): Stage D also embeds a card (a summary of what the chunk records, built from its profiles) as a
  second vector, the keyword index and the reranker see it; chunk text, answers and shown sources are unchanged. **Joint coverage**
  (`[features] joint_coverage`): in addition to the per-condition coverage of the doc, Stage 6 requires the named model / dataset /
  language to be recorded together by one result (benchmark suites count their member tasks), else the most specific condition is reported
  as not covered. **Weak-evidence flag withdrawn** when Stage 6 covers every named condition by recorded values. **Stage 7**: only conflicts in
  the named language; a text-only conflict needs a decimal / percentage in both sentences. **Stage 9**: list lead-ins and bare
  "label: number" lines are not claims, a claim without a metric is judged on number + model + setting, short "model: number" lines are
  checked, abbreviations ("vs.", "et al.") do not end a sentence. **Warm-up** at API start (`Pipeline.warm_up`), `[llm] keep_alive`
  (machine config) and `[llm] request_timeout`. Escalation is measured (no accuracy gain on Stage 6, +0.4 s per question): still ON.

## Open questions for the user (recommendation in brackets)
- A third "middle" complexity level? Doc and schema have two [keep two].
- Make Stages 7 and 9 decisions LLM-driven too? They already use an open-weights NLI model [leave as is].
- Joint condition coverage: BUILT 2026-10-01 (flag `joint_coverage`, ON): 22 -> 32 of 32 correct scope decisions. Beyond the doc [keep ON, needs the OK].
- Escalation (E2B -> 12B second opinion): measured, no gain on Stage 6, +0.4 s per question [decide after the team's job-B questions; flag `features.escalation`].
- Final LLM (12B / e4b / e2b) and per-agent models; which machine produces the paper's numbers.
- Orchestrator size (decided 2026-09-30 by the assistant under the user's delegation): E2B with the outcome-based escalation
  to 12B above; revisit after SciBERT is trained (comparisons then come out "complex" and the code guardrail decomposes them).

## Labelling plan (2026-10-01; labels are an ANSWER KEY for the evaluation, never training data)
The workbooks are BUILT (1 Oct): `python scripts/make_label_sheets.py` -> `data/labelling/` (8 files: A for Adarsh and Shashank, B for all four
with the planned 10 / 10 / 10 mix, C for Aditya and Tarun), guide `docs/labelling_guide.md`, scoring `scripts/score_labels.py` +
`scripts/eval_questions.py`. Rebuild the sheets if a paper is re-ingested (they snapshot the profile store).
Only one model is trained in the whole project: the Stage 1 SciBERT classifier, on LLM-written questions. Nothing is trained on
the team's labels (30 questions / 50 pairs are far too few, and training on them would spoil the test). Three jobs:
- **A. Profile check** (~300 random profiles from 10 papers: 6 easy - BERT, SQuAD 2.0, XNLI, XLM-R, GLUE, MuRIL - and 4 hard - T5,
  GPT-3, LLaMA, Llama 2): per field OK / WRONG / MISSING against the source table row -> the doc's field-level F1. ~1 min each.
- **B. Questions** (~30; 10 fully covered, 10 with one condition missing, 10 partly covered; each with intent + complexity,
  the conditions it names, corpus coverage looked up in a corpus map, and the expected behaviour). ~5 min each. Trap example:
  "How well do models perform on Kannada NLI?" IS covered (IndicXNLI); "XLM-R on XNLI for Kannada?" is NOT (XNLI has 15
  languages, none Kannada). Never assume coverage. Intent/complexity also test SciBERT on real questions.
- **C. Result pairs** (~50; two people label independently, a third settles; Cohen's kappa >= 0.6): EXPLAINED (tick which
  condition differs: dataset version/split, model size, language, setting, other) / GENUINE / NOT COMPARABLE, judged from the
  papers, not from the system's verdict. Real examples: human EM on SQuAD 82.3 vs 86.9 (v1.1 vs v2.0) = EXPLAINED; XLM-R XNLI
  average 83.6 (translate-train-all, XLM-R paper) vs 79.2 (zero-shot, mT5 paper's row) = EXPLAINED (setting).
Suggested split: Aditya + Tarun each label all 50 pairs; Adarsh + Shashank 150 profiles each; all four write 7-8 questions.
Due 2026-10-03 (evaluation runs 10-04/05). Gemini (browser) is allowed as an ASSISTANT only (draft questions, explain table
layouts, a second opinion after two humans labelled independently, pasted text only, prompt/date/model version recorded); it
must not be the labeller of record (same model family as Gemma -> correlated errors; the doc requires human kappa). The
official LLM baseline in the evaluation is local Gemma 12B with the same prompt. Never paste the architecture doc or the
system's answers/labels into a chat tool. The Excel sheets and the scoring scripts are built (see above).

## How to run
```
# after a reboot / long idle (a cold `import transformers` takes ~25 min on this PC otherwise):
.\.venv\Scripts\python scripts\prewarm_files.py                  # ~1 min
# Ollama (see gotcha above), then:
.\.venv\Scripts\python scripts\ingest.py [paper stems] [--force]     # do NOT run while the API/pipeline is running (cards are built automatically)
.\.venv\Scripts\python scripts\ask.py "Does BERT work well for Kannada question answering?"
.\.venv\Scripts\uvicorn cgrag.api.main:app --port 8000               # warms the models in the background (/health: warm.status); then: cd frontend; npm run dev
.\.venv\Scripts\python -m pytest
```
ChromaDB's persistent folder is single-process: never ingest and query at the same time.
Git push on this network sometimes fails once with "Recv failure: Connection was reset"; a plain retry works.

## Key files
- `config.toml` / `src/cgrag/config.py` — single source of truth for models, devices, paths, retrieval params, per-agent
  models (`[agents]`) and the ablation flags in `[features]` (orchestrator_agent, applicability_agent, applicability,
  contradiction, critic, profile_in_context, escalation, profile_guided_retrieval, joint_coverage; `[retrieval] use_cards`).
  Machine overrides go in git-ignored `config.local.toml`.
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

## Next steps (order of work; details in "RESUME HERE")
DONE 2026-09-30: ingestion frozen (`docs/ingestion_report.md`), SciBERT trained and installed (`docs/stage1_training_report.md`).
DONE 2026-10-01 (`docs/oct1_fixes_and_metrics.md`): critic, Stage 7 false conflicts, retrieval cards (threshold calibrated: -2.0 stays), joint
coverage, warm-up, escalation measured, labelling workbooks + scorers built. Left over on purpose: citation-labelled rows ("Devlin et
al. (2018)" = mBERT needs an alias step), group-label fragments in Llama 2, the two real stacked tables, the team's hand-check (labelling).
1. **Labelling** (see "Labelling plan"): send `data/labelling/*.xlsx`, collect the `_DONE` files by 3 Oct, `score_labels.py A/B/C`,
   `eval_questions.py` (Stage 1 on the team's questions + Stage 6 scope warnings; then the same with `--set features.escalation=false` and
   `--set features.joint_coverage=false` for the paper's ablation), then decide the escalation flag.
2. Open the UI in a browser (never done); test upload; show the joint-coverage warning and `warm.status` ("warming up") in the page. UI decision
   (user, 2026-09-30): keep it a simple Claude-style chat page for now and polish it late; it must keep showing the
   coverage bar and conflict badges (stage 10 of the doc). The page already sends chat history for follow-ups.
3. Build the evaluation sets and an ablation runner for the paper (flags above, incl. agent vs rules; `ablate_stage6.py` and
   `eval_retrieval.py` are the start); install RAGAS then (heavy dependencies; dry-run first).
4. Decide the final LLM / per-agent models and the machine for final numbers.
