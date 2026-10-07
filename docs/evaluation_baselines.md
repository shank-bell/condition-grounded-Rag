# Evaluation baselines (3 October 2026)

The architecture's evaluation plan ("How each claim will be tested") compares each new part with simple versions of the same job.
This file records what exists, how each baseline is defined (the paper must state these), and how the numbers are frozen so that
nothing is tuned to the team's answer key.

| Claim | Test set (the document's size -> ours) | Baselines in the document | Metric | Status on 3 Oct |
|---|---|---|---|---|
| Profiles are extracted correctly | 100 papers -> 10 papers, 300 profiles (job A) | MetaLead-style tuple extraction | field-level F1 | scorer built (`score_labels.py A`); **baseline extractor not built** |
| Stage 6 stops answers that go beyond the evidence | ~150 -> ~30 questions (job B) | Sufficient-Context autorater; abstention; plain RAG | extrapolation rate, scope-warning precision | **baselines built**; silver run done (4b); gold run when job B is back |
| Stage 7 separates genuine from explained conflicts | ~300 -> 50 pairs (job C), kappa >= 0.6 | plain NLI; LLM prompt on the DRAGged-into-Conflicts taxonomy | 3-class macro-F1 | **baselines built and frozen** (4a); scored when the gold is back |
| Answer quality is not reduced | Qasper, SQuAI -> **our own corpus** (a departure, section 3b) | pipeline without stages C, 6, 7 (+ the LLM alone) | value accuracy instead of answer-F1 / RAGAS | **harness built** (3b); silver run in 4c; gold run after job A |

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

## 3b. Answer quality (`scripts/eval_answers.py`, library `src/cgrag/evaluation/answers.py`) - a DEPARTURE from the document
The document tests "overall answer quality is not reduced" on Qasper and SQuAI with answer-F1 and RAGAS (an independent judge). Both need other papers
ingested (about 2 minutes per paper) and a judge from another model family than the answer model (only Gemma is installed), so this is scaled to our own
corpus. **Needs the user's OK before it is reported in the paper** (flagged in `CLAUDE.md`).
* **Questions:** result lookups with a known number - "What F1 does BERT-large get on SQuAD?" - one per (paper, model, dataset, version, language, metric) group,
  at most 3 distinct values per group (more would make a lucky number too likely), at most `n/8` per paper. **Silver** = built from the profile store (which
  may contain extraction errors). **Gold** = built only from the profiles the labellers judged ALL_OK in job A (`--verified-from A_*_DONE.xlsx`), so the answer
  is human-checked.
* **An answer is correct** when it states a recorded value of the group (up to rounding, +-0.05; citations like [3] are not numbers). Reported next to it:
  how many numbers an answer states (a longer answer has more chances to hit), the share of claims the critic supported, how many questions got a scope
  warning (every question is answerable, so each is a false one), seconds, and a paired exact sign test between systems.
* **Systems:** `full` (shipped); `no_stage_c67` = the pipeline without stages C, 6 and 7: `features.use_profiles = false` (no profile is used online: not in Stage 1's
  name vocabulary, not in the prompt, not in the critic), `applicability`, `contradiction`, `profile_in_context`, `profile_guided_retrieval`, `joint_coverage`,
  `escalation` off and `retrieval.use_cards = false`; `llm_only` = the answer model from memory, no retrieval.
* **Not an independent judge:** correctness is a number match, not an LLM verdict, so no judge bias; but it measures only lookup questions, not explanations.

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

### 4c. Answer quality, silver truth: 40 result-lookup questions (`scripts/eval_answers.py --silver`), 3 Oct
Questions are built from the profile store (3b). The same 40 were run twice, before and after the four fixes of 4d (same questions, same order).

| system | correct (before -> after) | numbers per answer | claims supported (own critic) | false scope warnings (of 40) | s / question |
|---|---|---|---|---|---|
| full pipeline | 40 -> 40 (100 %) | 3.05 -> 3.45 | 95 % -> 94 % | **8 -> 2** | 10.2 -> 10.8 |
| without stages C, 6, 7 | 26 -> 26 (65 %) | 2.58 -> 2.65 | 86 % -> 83 % | - | 7.0 -> 8.0 |
| LLM alone, from memory | 0 -> 0 (0 %) | 0.47 | - | - | 0.9 |

Paired exact sign test on the same questions: full vs without stages C, 6, 7: 14 questions only the full pipeline answered, 0 the other way, p = 0.0001; full vs
the LLM alone: 40 vs 0, p < 0.0001. The 14 misses of the pipeline without stages C, 6, 7: 6 said "the sources do not contain it" (retrieval did not surface the
table), 8 stated a different number (another setting, variant or paper), which exact-value matching counts as wrong. The LLM alone knows none of these table
cells (0 of 40): closed-book is not a competitive baseline for lookups. "Claims supported" is the system's own critic, which uses the profiles: not independent.

**Do not read this as "answer quality is not reduced".** The questions are built from rows the extractor read, so the full pipeline - which puts exactly those
recorded results in its prompt - is advantaged by construction, and tables the extractor missed produce no question at all. What the table shows: for results the
extractor captured, the profile-based stages find and state the number far more reliably than retrieval alone. The independent tests are `--gold-b` (the labellers'
answerable questions with their expected facts, written from the papers) and `--verified-from` (questions built from rows a human judged ALL_OK).

### 4d. What the answer-quality check exposed (3 Oct): four defects, all fixed, each with a unit test
The full pipeline gave a scope warning on 8 of the 40 answerable questions (all 8 were answered correctly). Reading them found:
1. **Fake further models** (Stage 1; mine, from the multi-entity work). The vocabulary route accepted ordinary words that a table row is labelled with - the store holds
   "Embeddings", "baseline", "memory", "human", "control" as models - so "GloVe embeddings" became the models "GloVe" and "embeddings", and Stage 6 warned that no
   source reports a model called "embeddings". A further model must now look like a name (a digit, a hyphen / underscore / plus, a capital after the first letter, or a
   capitalised word that is not the first of the question) and must not be a word of the first model's own name.
2. **Cut-off multi-word model names.** The LLM kept "GloVe" for "GloVe embeddings" (as it kept "GoldP" for "TyDi QA GoldP" on 1 Oct). The longest recorded multi-word
   name that contains the LLM's value and that the question writes literally now replaces it.
3. **A task that is part of the model's name.** The LLM also filed "embeddings" under task; a task value wholly inside the model's name is dropped.
4. **Stage 6 could not see rows labelled with "whose version" they are.** "Our BERT", "Google BERT", "Baseline (mT5-large)", "Avg. GloVe embeddings" are rows about BERT,
   mT5-large and GloVe embeddings, but a plain name matched only the same plain name. 149 of the 10,275 stored results carry such a prefix (70 BERT, 40 mT5-large: the
   names people ask about). `values_match` for models now ignores a leading google / our / published / baseline / vanilla / original / proposed / avg / average word when a
   separator follows it (`GoogLeNet` is not "Google" + "Net"). `model_family`, and with it which two results Stage 7 pairs, is unchanged on purpose: the frozen Stage 7 key
   re-classifies identically (50 of 50).

Effect on the same 40 questions: **false scope warnings 8 -> 2**, accuracy unchanged. The two left are artefacts of building silver questions from raw store rows (a task
literally named "Anaphora/Coreference (Coref)"; a model written "Baseline (mT5-large)" that the LLM splits). Regression checks on the fixed code: the 32-question Stage 6
ablation stays at **32/32** (after the Stage 1 fixes, and again after all four), the 12-question scorer dry run has 0 misses (model detection precision and recall 1.0), the
8 end-to-end questions of `oct1_fixes_and_metrics.md` section 6 behave as before, 193 unit tests pass.
Honest limits: the fixes were found on silver questions before any gold label existed, so they are not tuned to the answer key; whether other name variants cause false
warnings on the team's questions is what the gold run will show. The Stage 6 table of 4b was produced before these fixes: it is re-run from one commit at the freeze.

## 5. Caveats the paper must carry
* Both LLM baselines and the judge-like components are the same model family as the answer (Gemma), the project may not call external APIs.
* Plain RAG's extrapolation rate is 100 % by construction (it has no way to warn); the informative comparisons are the abstention baseline
  and the autorater.
* The silver questions are built from the profile store, so they cannot expose a false warning caused by a *missing* profile (a table the
  layout model missed); the gold questions, written from the papers, can.
* 50 pairs and ~30 questions are small: report counts next to every percentage.
* The answer-quality questions are drawn from rows the extractor captured (selection bias in favour of the full pipeline), and "correct" means "states a recorded
  number": an answer that cites a real number from another setting counts as wrong, a longer answer has more chances to hit (numbers per answer is reported).

## 6. Not built yet (and the default I would take)
* **Answer quality (claim 4):** built as a scaled version on our own corpus (3b); it is a departure from the document (Qasper / RAGAS) and needs the
  user's OK before it is reported. Qasper / SQuAI would need other papers ingested and RAGAS a judge from another model family.
* **RAGAS:** not installed (checked 3 Oct: `pip show ragas` finds nothing) and never run. The architecture removed it from the runtime (the NLI claim critic replaced it) and kept it for
  offline evaluation, with "an independent judge". Only Gemma (12B, 4B, 2B) is local and no external API may be called, so a judge from another model family would have to be pulled through Ollama
  (evaluation only, about 5-9 GB; it is not part of the system). Plan for 6 Oct, if the labels have arrived: (A) install RAGAS and run faithfulness and answer relevancy on the same 40 questions
  with Gemma 12B as judge, labelled "not independent"; (B) with the user's OK, repeat with a judge of another family. The judge-free number-match test of 3b stays the primary metric.
* **Extraction baseline (claim 1).** A MetaLead-style extractor (model, dataset, metric, value only, no conditions) on the job-A tables would show
  that the condition fields cannot be filled without the profile extractor. Not built; only worth it if time remains after the scoring.
