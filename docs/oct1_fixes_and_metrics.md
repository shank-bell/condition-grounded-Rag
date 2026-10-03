# 1 October 2026: the five open items, what was found, and the numbers

Everything below was measured on this PC (RTX PRO 4000, 24 GB) on the 28-paper corpus (1,385 chunks, 10,275 profiles) with
`gemma4:12b` for the answer and `gemma4:e2b` for the two agents. Test suite: **175 tests pass**. The measurement scripts are in
`scripts/` (named below) so every number can be reproduced.

## What was asked, in one table
| Open item (from the 30 Sep report) | What it turned out to be | Fixed? | Evidence |
|---|---|---|---|
| Critic marked three IndicBERT numbers "unsupported" | three separate critic gaps (below) | yes | Kannada NLI answer: 3 of 5 claims rejected -> **6 of 6 supported** |
| "DistilBERT vs BERT-base on GLUE" gave "not comparable (profile missing)" | **my diagnosis on 30 Sep was wrong**: the GLUE table was extracted (31 profiles). The note came from NLI calling two unrelated "We compare X with ..." sentences contradictory | yes | no conflict note any more; 12 claims, 0 unsupported |
| Kannada question kept only 1 chunk on the first retrieval | the table that holds the answer was **never retrieved** (header says `kn`, caption says `IndicXNLI`); the reranker scored it -10.8 | yes (retrieval cards) | hit@5 69 % -> 90 % (all), 22.5 % -> 60 % (language + task questions) |
| Escalation's latency cost and gain unmeasured | measured: **no accuracy gain, +0.4 s per question on average** (+3.5 s on each of the 4 of 32 questions where it fires) | measured | table 5 |
| First question after a restart takes 30-106 s | cold models + a slow cold Python start on this PC | yes | first question **38.8 s -> 10.0 s**; Python start 25 min -> 1 min |

Two findings that were not on the list but matter more than any of them: **Stage 6 almost never warned when a language was missing**
(only 5 of 12 "should warn" questions), fixed by a joint-coverage check (5/12 -> **12/12**, section 4), and the labelling workbooks
are built (section 8).

## 1. Critic (Stage 9): why three numbers were "unsupported"
Diagnosed with `scripts/debug_critic.py` (prints, for every rejected sentence, the recorded results that carry its numbers and the
test each one fails). Root causes, all fixed in `critic.py` / `text.py`, each with a unit test:
1. A list lead-in ("... the following models achieved these accuracy scores:") is not a claim: sentences ending in ":" are skipped.
2. A list item such as `IndicBERT + Samanantar: 68.2 (test set)` matches a recorded result on number, model and setting but was
   rejected because the **metric name sits in the lead-in line**, not in the bullet. When a claim names no metric, number + model +
   setting decide; when it names one, it must match the recorded metric.
3. Short numeric lines ("MuRIL: 67.8", 2 words) were **never checked at all** (a silent gap). They are now checked when they name a
   system; a bare condition value under a model heading ("CoLA: 56.3") has no subject and is skipped.
4. The sentence splitter broke at "vs." / "et al." / "e.g." / "Fig.", so the critic judged fragments: abbreviations no longer end a sentence.
5. (evening) A sentence that says a result is **missing** ("No results are reported for Kannada for any of the models.", "GPT-4 was not
   evaluated on XNLI.") is what the scope warning asks the answer to say, but the critic judged it like a claim, found no source that
   entails it, and regenerated the whole answer for nothing (+3 s, and the claims panel showed two "unsupported" claims that were
   correct). Such sentences now join the other "what the sources do not contain" sentences the critic already skipped. **Limit:** a
   *wrong* absence statement ("no results for X" when X is recorded) is not caught either; Stage 6's coverage is the check for that.

## 2. Stage 7: the false "conflict" and the wrong-language conflicts
* Text-only disagreement (no recorded numbers) now needs a **reported result in both sentences**: a decimal or a percentage. A year in a
  citation ("Sun et al., 2020") or the 6 in "BERT6" is not a result. Before: two introductions ("We compare TinyBERT with ...",
  "We compare our MobileBERT with ...", from two other papers) were flagged contradictory by NLI at 0.98.
* A question that names a **language** only counts conflicts between results in that language (a question about Kannada was shown a
  disagreement between two Hindi results). Results with no recorded language count as English.
* Stage 7 still compares with the strict `values_match` (a GLUE score is never compared with a CoLA score).

## 3. Column-wise tables (the "11 of 350")
Looked at the raw layout of each flagged table. Only **2 are real stacked / side-by-side merges** (RAG p6, DistilBERT p3 Tables 2+3,
which hold parameter counts and timings). The others are not column-wise at all: TinyBERT p11 is an ordinary table whose headers wrap
over three lines (already parsed correctly), GPT-3 p19 is two blocks with a repeated header row, Llama 2 p6 mixes a figure with a table,
Llama 2 p61 is prose examples. **Decision: no un-stacker** (limitation, documented in `ingestion_report.md`). Also removed: 30 profiles
whose metric was a run of letters from a mis-read header (`mniorpasasatsdtateuravg`): profiles 10,305 -> **10,275**.

## 4. Retrieval, the warning decision, and what each step is worth
### 4a. Retrieval cards (Stage D / 4 / 5)
A **card** is a one-paragraph summary of what a chunk records, built from its own profiles ("Reported results - tasks: natural language
inference; datasets: IndicXNLI; languages: Assamese, Bengali, ..., Kannada, ...; models: ...; metrics: accuracy"). It is embedded as a
second vector per chunk, added to the keyword index, and shown to the reranker in front of the chunk text. The chunk text, the answer
and the sources shown are unchanged. 284 of 1,385 chunks have a card (mean 245 characters); building them takes 16 s, no LLM.
`scripts/eval_retrieval.py`: 190 questions made from the profile store (150 "model + dataset (+ language)", 40 "language + task", e.g.
"How well do models perform on Kannada NLI?"), relevant chunk = a chunk with one profile that records all the question's conditions.

| | baseline | with cards |
|---|---|---|
| all: hit@20 (a relevant chunk among the 20 candidates) | 82.1 % | **95.3 %** |
| all: hit@5 after the reranker | 68.9 % | **90.0 %** |
| all: MRR after reranking | 0.556 | **0.789** |
| model + dataset questions: hit@5 / MRR | 81.3 % / 0.670 | **98.0 % / 0.872** |
| **language + task questions: hit@20 / hit@5 / MRR** | 40.0 % / 22.5 % / 0.126 | **82.5 % / 60.0 % / 0.477** |
| 20 questions about things NOT in the corpus, flagged "weak evidence" | 65 % | 70 % |

### 4b. The reranker threshold (-2.0) was already right
Sweep over 10 thresholds on the same questions (with cards): the hit rate is 90.5 % at -2.0 and 94.7 % at -6, but the share of
out-of-corpus questions flagged weak falls from 70 % to 45 %. **-2.0 is kept** (best trade-off).

### 4c. Joint coverage: the finding that mattered
Stage 6 checks each named condition separately. "XLM-R on MLQA for Kannada" has XLM-R (many papers), MLQA (7 languages) and Kannada
(in Indic papers), so every condition looked covered although no result records them together. The better retrieval of 4a made it
worse (more Kannada chunks found). New guardrail (`[features] joint_coverage`, `applicability.py`): when the question names two or
three of **model, dataset, language**, the profile store must contain **one result that records them together** (a benchmark suite such
as GLUE counts its member tasks: CoLA, MRPC ...). If it does, those chunks are added to the evidence; if not, one condition is reported
as not covered - the most specific one the others are recorded without (language, then dataset, then model) - and the warning says what
IS recorded: "No source reports language = Kannada together with model = XLM-R, dataset = XNLI. For these, the sources record:
English, Spanish, French, German, Bulgarian, Russian ...".

### 4d. Ablation (`scripts/ablate_stage6.py`, 32 questions, Stages 1-6)
12 "covered" (model + dataset + language triples with a recorded result: no warning expected), 12 "missing" (the same models and
datasets with a language the benchmark does not have: XNLI has 15 languages, XQuAD 11, MLQA 7; a warning naming the language expected),
8 "language + task" (a result exists: no warning expected). Truth comes from the profile store and from facts about the benchmarks,
**not** from human labels (labelling job B is the human version).

| configuration | wrongly warned (covered 12) | correct warning (missing 12) | wrongly warned (language+task 8) | **correct decisions (of 32)** | s / question |
|---|---|---|---|---|---|
| baseline (the pipeline before 1 Oct) | 0 | 5 | 3 | **22** | 3.1 |
| + retrieval cards | 0 | 3 | 0 | 23 | 3.1 |
| + profile-guided retrieval | 0 | 3 | 0 | 23 | 3.1 |
| + joint coverage | 0 | **12** | 0 | **32** | 3.4 |
| + E2B -> 12B escalation (second opinion) | 0 | 12 | 0 | 32 | 3.8 |

(rows 2-3 from the first run, rows 1, 4, 5 from the final run with all fixes; the baseline reproduced exactly: 22/32 both times.)
**Escalation:** it fired on 4 of the 32 questions, changed no decision, and cost +3.5 s each (Stage 6 0.36 s -> 0.76 s on average).
Before the joint guardrail it even flipped one correct warning to "covered" (the bigger model is more permissive). The flag stays ON
because it was the user's design; the data say it is not earning its cost for Stage 6. **Decide after the team's job-B questions:**
`python scripts/eval_questions.py eval/labels/questions_gold.jsonl --set features.escalation=false --tag no_escalation`.

Caveat on 4d: the "covered" questions are built from the store, so they cannot reveal a **false warning caused by a missing profile**
(a table the layout model missed: the joint check trusts the store). Job B's "covered" questions are written from the papers, so they will.

### 4e. "Weak evidence" no longer contradicts the profiles
The cross-encoder cannot read results tables (even with the card it scores the Kannada NLI table -3.9). When Stage 6 reports that every
condition named in the question is recorded in the kept passages, the weak-evidence flag is withdrawn; before, the answer ended with a
bogus "The SCOPE WARNING indicates ..." line although the numbers were right (IndicBERT+Samanantar 74.7, MuRIL 74.0, XLM-R 71.5, mBERT 58.6).

### 4f. Questions that name several systems or languages (evening; part of `[features] joint_coverage`)
**Found** while dry-running the labelling scorer on 12 invented questions: "Compare mBERT and GPT-4 on XNLI" got **no scope warning**.
Stage 1 returned one model (mBERT), so Stage 6 never looked at GPT-4, which no paper reports. Job B has such questions by design
("partly covered", 10 of 30), so each would have counted as a missed warning.
**Change.** `QueryConditions` gets `other_models / other_datasets / other_languages` (and `extras()`; `specified()` is unchanged, so no
other stage is touched). Stage 1 fills them from the LLM and from names found literally in the question (the store's model names, the
language list); a value must occur in the question. Stage 6 (`_extras`) checks each further value like the joint check: one result must
record it together with the other named model / dataset / language. Covered ones add their best two chunks to the evidence (every
compared system gets a source); uncovered ones become "not covered" conditions and the warning names them and says what IS recorded.
**Three problems the regression run itself exposed (all fixed, each with a unit test):**
1. With the QA benchmarks (XQuAD, MLQA, TyDi QA) the 12B model files the **only** dataset under `other_datasets` and leaves `dataset`
   empty: 8 of the 32 ablation questions got the same dataset twice (harmless for the verdict, wrong analysis). Fix: when the plain
   field is empty, the value named first in the question becomes the first one; the prompt now says a single value never goes in an
   `other_*` list.
2. With the new prompt the model abbreviated "TyDi QA GoldP" to "GoldP", which matches nothing in the store (1 false warning, 31/32). A
   dataset name the LLM cut short is replaced by the full name the question writes when the store has it.
3. For "mBERT, XLM-R and GPT-4 on XNLI for Hindi and Kannada" the answer rightly said "No results are reported for Kannada ..." and the
   critic marked it unsupported (section 1, item 5).
**A validity slip of mine, corrected:** the multi-entity example in the Stage 1 prompt was word for word the question I used to test it,
so the first end-to-end run proved nothing. The example is now unrelated (ALBERT / ELECTRA on RACE and SQuAD for English and Tamil) and
the tests use questions that are not in the prompt.

| check (final code) | result |
|---|---|
| `ablate_stage6.py --configs "+joint"` (32 questions, one entity of each kind) | **32/32 correct**: 0 of 12 covered wrongly warned, 12 of 12 missing-language questions warned and name the language, 0 of 8 language+task wrongly warned; **0 questions got a further entity**; 3.2 s per question (3.4 s in the run before: no change) |
| dry run of `eval_questions.py` on 12 invented questions (incl. "Compare mBERT and GPT-4 on XNLI") | intent 12/12, complexity 12/12, scope warning 4 of 4 expected warnings given and **none wrongly**, the right missing condition named 4/4 (before the change: "Compare mBERT and GPT-4 on XNLI" had no warning) |
| "Compare mBERT, XLM-R and GPT-4 on XNLI for Hindi and Kannada" | further entities found: XLM-R, GPT-4, Kannada; **coverage 67 %** (4 of 6): GPT-4 and Kannada reported missing, warning says what IS recorded; 2/2 claims supported, no regeneration, 11.5 s (before the critic fix: 3/5 claims, 13.7 s) |
| "How well do models perform on Kannada and Tamil NLI?" | Tamil found as a further language and covered (100 %), 16/16 claims supported |
| "Does BERT-base or GPT-3 do better on SQuAD v2.0?" | GPT-3 found, covered (100 %), 3/3 claims supported |
| "How do mT5 and XLM-R compare on XQuAD for Arabic and Thai?" | warns that no source records XLM-R on XQuAD in Arabic (**may be a false warning**: the mT5 paper's per-language XQuAD tables are among the ~15 tables the layout model missed, which is exactly the caveat of 4d); the answer then recites mT5 *ablation* rows (span length, dropout ...) as if they were results and the critic rejects both claims (0/2). An extraction weak spot (ablation-table rows stored as models), not the new code; not investigated further |
Not covered: a second dataset is found only when the LLM names it ("GLUE and SQuAD" kept SQuAD); the vocabulary route is not used for
datasets because the store's dataset names are too noisy ("Dev Set SST-2") and would raise false warnings.

## 5. Start-up and latency (`scripts/measure_first_question.py`, `scripts/prewarm_files.py`)
* **First question after a restart: 38.8 s -> 10.0 s.** `Pipeline.warm_up()` loads the embedder, reranker, NLI, SciBERT and both Ollama
  models once (19.6 s, torch models one after another, Ollama models in parallel). The API runs it in a background thread at start:
  `/health` answers in 1.7 s with `"starting"`, the warm-up finishes after 30 s, the first real `/query` then takes 6.2 s.
  (My first version loaded the torch models in parallel threads; `from_pretrained` sets a process-wide default dtype, so one model ended
  up half fp16 / half fp32 and crashed an ablation run. Fixed and documented in the code.)
* `[llm] keep_alive = "2h"` (this PC) and `[llm] request_timeout = 240 s` (a wedged Ollama request used to block a job for 15 minutes).
* **A cold Python start took ~25 minutes this morning.** Cause found with `py-spy`: every first read of a file costs ~0.25-0.3 s on this
  PC (same on C: and D:, a warm read ~0 ms; some scanner checks each file once) and `import transformers` opens thousands of files in
  series. The delay is per-file latency, so reading files in parallel removes it: `python scripts/prewarm_files.py` touches 26k files in
  52 s, then the imports take 6 s. **Run it first after a reboot or a long idle.** (It probably also explains the laggy terminal.)
* Warm questions, Stages 1-10, five key questions: median **10.9 s** (7.3-16.4 s); the median was 8.3 s on 30 Sep, the difference is
  more regenerations by the (now stricter) critic and the larger scope-warning text.

## 6. End-to-end check of the five key questions (final code)
Re-run in the evening with the final code (`scripts/smoke_questions.py`, all stages; the model's wording varies from run to run, so claim
counts move by a sentence or two between runs).
| Question | Result |
|---|---|
| How well do models perform on Kannada NLI? | answered from IndicXNLI (6 models), coverage 100 %, 6/6 claims supported, no stray warning, no further entity |
| What accuracy does XLM-R get on XNLI for Kannada? | **warns**: no source records language = Kannada together with XLM-R and XNLI; says what is recorded (English, Spanish, ...); 5/5 claims supported |
| Compare DistilBERT and BERT-base on GLUE | BERT-base now recorded as a further model and checked jointly with GLUE: coverage 100 % (3 checks), no false conflict, 8/8 claims supported, 9.3 s (an earlier run of the same code: 6/6 after one regeneration, 17.8 s) |
| What F1 does BERT-large get on SQuAD v2.0? | two EXPLAINED conflicts (dataset version); two claims flagged (5/7): "89.1 F1" and "83.1 F1" - the store has wrong BERT-large rows in the ALBERT paper (an extraction error to check in job A) |
| What accuracy does mBERT get on XNLI for Hindi? | covered, one NOT_COMPARABLE note between two papers' Hindi numbers, 3/3 claims, 7.7 s |
Latency of that run: first question 27.5 s (this script does not call the warm-up; the API does), the other eight questions median
10.4 s (7.7-21.3 s). The multi-entity questions are in section 4f.

## 7. Departures from the architecture added today (all behind flags; need the user's OK, delegated to the assistant's judgement)
1. **Retrieval cards** (`[retrieval] use_cards`): Stage D embeds a second vector per chunk from a card built from its profiles; Stage 4
   searches it and the keyword index sees it; Stage 5 reads the card in front of the chunk text. Order of the offline stages unchanged
   (profiles are extracted before embedding, as in the doc).
2. **Joint coverage** (`[features] joint_coverage`): an addition to Stage 6's per-condition coverage; the coverage bar and the per-condition
   checks stay as in the doc.
3. **Weak-evidence flag withdrawn** when Stage 6 covers every named condition by recorded values.
4. **Stage 7**: language filter for conflicts; text-only conflicts need a reported result in both sentences.
5. **Stage 9**: the critic changes of section 1 (including that "no results are reported for X" sentences make no claim).
6. Escalation + profile-guided retrieval (built 30 Sep) - measured above.
7. **Several systems / languages in one question** (section 4f; part of `[features] joint_coverage`): Stage 1 may return further models /
   datasets / languages and Stage 6 checks each one jointly with the other named conditions.

## 8. Labelling kit (built, tested, not yet used by the team)
`python scripts/make_label_sheets.py` -> `data/labelling/` (git-ignored): 8 workbooks + private keys; guide `docs/labelling_guide.md`.
* **A** profile check: `A_profile_check_Adarsh.xlsx`, `..._Shashank.xlsx`: 300 random profiles of 10 papers (6 easy, 4 hard), 150 each + 20
  shared for an agreement check; per row: arXiv link to the exact page, the table caption + header + the row with the number marked,
  one verdict dropdown (ALL_OK / SOME_WRONG / NOT_A_RESULT / CANNOT_CHECK) and per-cell WRONG / MISSING marks only when needed.
* **B** questions: `B_questions_<Aditya|Adarsh|Shashank|Tarun>.xlsx` (8, 8, 7, 7 rows; planned mix 10 covered / 10 one missing / 10 partly
  covered) with five lookup sheets (Papers, Datasets, Dataset x Language, Models, Model x Dataset) built from the extractor's output.
* **C** pairs: `C_result_pairs_Aditya.xlsx`, `..._Tarun.xlsx`: the same 50 result pairs in each person's own order (365 were available;
  7 genuine / 29 explained / 14 not-comparable by the system, shown to nobody), both sources with links, verdict + which condition differs.
* Scoring: `scripts/score_labels.py {status,A,B,C}` (field-level P/R/F1, gold questions, Cohen's kappa, adjudication sheet, Stage 7
  accuracy / per-class F1 / confusion) and `scripts/eval_questions.py` (runs the system on the labelled questions; Stage 1 + Stage 6 scores;
  `--set` for ablations). 15 tests including full round trips (build -> fill in -> parse -> score).
* Not verified: LibreOffice is not installed on this PC, so the workbooks were not recalculated or opened in Excel here. The only formulas
  are three trivial progress counters (`COUNTA` / `ROWS`) and the workbooks ask Excel to recalculate on open.
