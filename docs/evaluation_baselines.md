# Evaluation baselines (3 October 2026)

The architecture's evaluation plan ("How each claim will be tested") compares each new part with simple versions of the same job.
This file records what exists, how each baseline is defined (the paper must state these), and how the numbers are frozen so that
nothing is tuned to the team's answer key.

| Claim | Test set (the document's size -> ours) | Baselines in the document | Metric | Status on 3 Oct |
|---|---|---|---|---|
| Profiles are extracted correctly | 100 papers -> 10 papers, 300 profiles (job A) | MetaLead-style tuple extraction | field-level F1 | scorer built (`score_labels.py A`); **baseline extractor not built** |
| Stage 6 stops answers that go beyond the evidence | ~150 -> ~30 questions (job B) | Sufficient-Context autorater; abstention; plain RAG | extrapolation rate, scope-warning precision | **baselines built**; silver run done (4b); gold run when job B is back |
| Stage 7 separates genuine from explained conflicts | ~300 -> 50 pairs (job C), kappa >= 0.6 | plain NLI; LLM prompt on the DRAGged-into-Conflicts taxonomy | 3-class macro-F1 | **baselines built and frozen** (4a); scored when the gold is back |
| Answer quality is not reduced | Qasper, SQuAI -> *not decided* | pipeline without stages C, 6, 7 | answer-F1, RAGAS (independent judge) | **not built** (section 6) |

## 1. What is frozen, and how
`src/cgrag/evaluation/freeze.py` stamps every evaluation output with the commit, the list of uncommitted files and the time. The Stage 7
baselines were produced on 3 Oct 05:52 UTC (11:22 local) at commit `5c9fc56`; the only uncommitted files then were additive (a response
field, the scorer, the new scripts), none of them touches Stage 7. **The system's verdicts (`private/job_C_key.json`, built 1 Oct) and the
baselines' verdicts exist before any label does.** Later fixes are reported separately, never merged into these numbers.
The LLM runs at temperature 0: re-running gave identical class counts.

## 2. Stage 7 baselines (`scripts/run_pair_baselines.py` -> `data/labelling/private/job_C_baselines.json`, git-ignored)
All three systems judge the same 50 pairs the labellers judge (the system's own verdict is hidden from them).
* **plain_nli** - DeBERTa-v3 NLI (the model Stage 7 uses for text-only conflicts) on the two results written as statements ("ELMo obtains
  75.2 accuracy on QNLI (dev, English)."), both directions, probabilities averaged. Contradiction -> GENUINE; neutral or entailment ->
  NOT_COMPARABLE. A plain NLI detector has no notion of a difference that conditions explain, so **it can never predict EXPLAINED**.
* **llm_taxonomy** - the same local Gemma 12B as the answer, zero-shot, temperature 0, with a prompt built on the five categories of
  "DRAGged into Conflicts" (Cattan et al. 2025), shown exactly what the labellers see (caption, header, the row with the number marked).
  Mapping: complementary information -> EXPLAINED (its list of differing conditions is kept); conflicting research outcomes, outdated
  information, misinformation -> GENUINE; no conflict (or an unrecognised answer) -> NOT_COMPARABLE.
* **stage7_rerun** - Stage 7's own classifier run again on the stored profiles: it agrees with the key on **50 of 50** pairs, so the key
  is what the current code says.

| frozen verdicts on the 50 pairs | GENUINE | EXPLAINED | NOT_COMPARABLE |
|---|---|---|---|
| Stage 7 (the key) | 7 | 29 | 14 |
| plain NLI | 48 | 0 (by construction) | 2 |
| LLM on the taxonomy | 13 | 33 | 4 |

These are class counts, **not accuracy**: whether any of them is right is decided only by the gold (two independent labellers, a third for
disagreements). `python scripts/score_labels.py C --first ... --second ...` prints Stage 7 and both baselines side by side
(accuracy, macro-F1, F1 per class) once the labels are in.

## 3. Stage 6 baselines (`scripts/eval_scope_baselines.py`)
Each question runs once **without** Stage 6 (plain RAG: Stages 1-5 of the shipped system, its passages and its weak-evidence flag) and once
per Stage 6 configuration. No answer is generated: the decision does not need one. What "warns" means:
* **plain_rag** - never. No Stage 6 and no abstention, so every should-warn question is answered as if the evidence covered it.
* **abstain_on_weak_retrieval** - Stage 5's own flag (no retrieved passage above the reranker threshold). `QueryResponse.retrieval_weak`
  carries it (added for this; additive, the UI ignores it).
* **sufficient_context** - an LLM autorater in the style of Joren et al. (2024) judges whether the plain-RAG passages (the top five, each cut
  to 1,800 characters) are enough to answer the question as asked; "not sufficient" = warns. It runs on the local Gemma 12B; the published
  autorater used a far larger model, so **this baseline is probably weaker than the published one** and the paper must say so.
* **stage6_30sep / stage6_joint / stage6_full** - Stage 6 as of 30 Sep (per-condition coverage only), with cards + profile-guided retrieval +
  joint coverage, and the shipped configuration (+ the E2B -> 12B second opinion).

Metrics: precision, recall and F1 of "warns" (positive = the question should get a warning); **extrapolation rate** = the share of
should-warn questions answered with no warning at all (= 1 - recall); false-warning rate = the share of should-not-warn questions that got
one; "condition named" = of the correct warnings, how many name a condition the truth names (only Stage 6 names one). Truth: **silver** =
derived from the profile store and from facts about the benchmarks (XNLI has 15 languages, none of them Kannada ...), not from people;
**gold** = the labellers' job-B answers (`--gold eval/labels/questions_gold.jsonl`).

## 4. Results so far
### 4a. Stage 7: see the table in section 2 (frozen; accuracy waits for the labels).
### 4b. Stage 6, silver truth: 32 questions (12 should warn, 20 should not), run 3 Oct 11:29-11:42 at commit `5c9fc56`
Per-question rows for every system: `data/index/scope_baselines_silver.json` (git-ignored). Decisions of the 32 questions: 12 "missing" (a language the
benchmark does not have), 12 "covered" (a model + dataset + language triple with a recorded result), 8 "language + task" (e.g. Kannada NLI).

| system | warnings given | needed warnings caught (of 12) | false alarms (of 20) | precision | recall | F1 | extrapolation | right condition named | s / question |
|---|---|---|---|---|---|---|---|---|---|
| plain RAG (never warns) | 0 | 0 | 0 | n/a | 0 % | 0 % | 100 % | - | 4.1 |
| abstain on weak retrieval | 1 | 0 | 1 | 0 % | 0 % | 0 % | 100 % | - | 4.1 |
| Sufficient-Context autorater (Gemma 12B) | 23 | 11 | 12 | 48 % | 92 % | 63 % | 8 % | - | 12.7 |
| Stage 6 as of 30 Sep | 10 | 6 | 4 | 60 % | 50 % | 55 % | 50 % | 5 of 6 | 3.5 |
| Stage 6 + cards, guided search, joint coverage | 12 | 12 | 0 | 100 % | 100 % | 100 % | 0 % | 12 of 12 | 3.2 |
| Stage 6 as shipped (+ E2B -> 12B second opinion) | 12 | 12 | 0 | 100 % | 100 % | 100 % | 0 % | 12 of 12 | 3.8 |

(seconds per question = Stages 1-5/6 plus the method's decision, no answer; the autorater's 12.7 s = 4.1 s of plain RAG + the LLM call.)
The 30 Sep row reproduces the 1 Oct ablation (`oct1_fixes_and_metrics.md` 4d): 22 of 32 decisions right (6 + 16).

**Why each method fails** (read off the per-question rows):
* **Abstain on weak retrieval** misses all 12: the search still returns relevant-looking passages (XNLI, MLQA ...) for a language those benchmarks do
  not contain. Relevance is not coverage. Its single warning is a false alarm (Kannada NLI, a table the cross-encoder cannot read).
* **Autorater** catches 11 of 12 but raises 12 false alarms on the 20 safe questions: 7 on "covered" questions (it asks for a metric the question words
  loosely - "accuracy" for F1/EM benchmarks - or for per-language rows that the five truncated passages lack) and 5 on language + task questions (the
  plain-RAG passages simply do not hold the Kannada / Tamil / Malayalam table that the store knows about: a retrieval failure that Stage 6 repairs with
  a profile-guided search). Its one miss is the trap the project exists for: "XLM-R on XNLI for Marathi" - it took IndicXNLI's Table 16 ("mr" 69.0) for XNLI.
* **Stage 6 on 30 Sep** missed the 6 "XLM-R / mBERT on MLQA for <a language MLQA lacks>" questions (each condition is recorded somewhere, but no result records
  them together) and gave 4 false alarms on language + task questions (retrieval had not found the table). Joint coverage fixes the first, cards + guided search the second.

**Caveat that matters: do not quote the 100 % as "the result".** The silver questions are generated from the same profile store that Stage 6's joint check consults
(and the should-warn ones from facts about benchmark languages), so a perfect Stage 6 score is partly built in, and the autorater is handicapped by seeing only five
truncated passages. What the table does show is the direction and the failure modes of the simple methods. The **gold run** - the team's job-B questions, written from the
papers - is the test that can expose Stage 6's real false-warning rate, including warnings caused by tables the extractor missed.

## 5. Caveats the paper must carry
* Both LLM baselines and the judge-like components are the same model family as the answer (Gemma), the project may not call external APIs.
* Plain RAG's extrapolation rate is 100 % by construction (it has no way to warn); the informative comparisons are the abstention baseline
  and the autorater.
* The silver questions are built from the profile store, so they cannot expose a false warning caused by a *missing* profile (a table the
  layout model missed); the gold questions, written from the papers, can.
* 50 pairs and ~30 questions are small: report counts next to every percentage.

## 6. Not built yet (and the default I would take)
* **Answer quality (claim 4).** The document names Qasper and SQuAI; both need other papers ingested (about 2 minutes per paper on the 12B
  extractor) and RAGAS needs a judge from another model family than the answer (only Gemma is installed here, so it would not be independent).
  Default: a scaled version on our own corpus - store-derived questions with a known numeric answer, answer-correct = the recorded value appears
  in the answer, full pipeline vs the pipeline without the profile extractor's output, Stage 6 and Stage 7 (`profile_in_context`,
  `applicability`, `contradiction` off). This is a departure from the document (Qasper / RAGAS) and needs the user's OK before it is reported.
* **Extraction baseline (claim 1).** A MetaLead-style extractor (model, dataset, metric, value only, no conditions) on the job-A tables would show
  that the condition fields cannot be filled without the profile extractor. Not built; only worth it if time remains after the scoring.
