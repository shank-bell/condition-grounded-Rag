# Stage 1 (SciBERT intent + complexity classifier): training report, 2026-09-30

The architecture's Stage 1 is one SciBERT (`allenai/scibert_scivocab_uncased`) with two heads: **intent** (factual, method,
result, comparison, survey) and **complexity** (simple, complex). It is the only model trained in this project. Spec, targets,
monitor-agent prompt and the iteration log: `docs/stage1_campaign.md`. Installed at `data/index/query_classifier/`
(git-ignored, 419 MB; `scripts/install_query_classifier.py`), loaded automatically by Stage 1 (`scibert+llm`); the LLM still reads
the conditions (language, dataset, model ...) of the question.

## Result
| Set | n | Intent acc / F1 | Complexity acc / F1 | Both right |
|---|---|---|---|---|
| val_clean (held-out slice of the training data) | 494 | - / 0.975 | - / 0.977 | |
| val_llm (its LLM-written part, no templates) | 332 | - / 0.963 | - / 0.967 | |
| dev A (68, used for decisions) | 68 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 |
| dev B (39, errors read once after iteration 1) | 39 | 0.974 / 0.976 | 1.000 / 1.000 | 0.974 |
| **dev C (48, written after iteration 1, never used for a decision)** | 48 | **1.000 / 1.000** | **1.000 / 1.000** | **1.000** |
| dev C typed like a real user (typo / lower case / chatty prefix, x5) | 240 | 0.983 / 0.985 | 1.000 / 1.000 | 0.983 |

All six targets of the campaign are met by the installed model (val_clean F1 >= 0.97 / 0.95, val_llm F1 >= 0.95 / 0.95, dev-A
accuracy >= 0.92 / 0.90). The four errors on the typo probe are typos in the key word ("Copmare", "corpoar").

**What "perfect" means here.** The labels are silver (an LLM wrote and a second LLM pass confirmed them), and the dev sets were
written by the assistant, so 100 % on dev C means "agrees with the rubric as the assistant reads it", not a human gold standard.
The remaining ~3 % validation errors are boundary cases where the label itself is arguable ("why does SBERT perform worse than
BERT ..." is method or comparison; "what are the main datasets for X" is survey or factual). The real test is the team's own
30 questions (labelling job B, 2026-10-01), which score this classifier on questions nobody at the model level has seen.

## What was done (3 iterations, 17:06-18:10, no human input)
1. **Data.** 8 classes of the rubric (a comparison and a survey question is always complex). LLM-written questions from real
   passages of the 28 papers (Gemma 12B; plain / informal / terse / multipart styles; 5,687 questions in two rounds), 1,436-1,600
   template questions with names taken from the profile store (labels correct by construction), 15 hand-written seeds. A **blind
   relabel** (12B sees only the question and a rubric) keeps an LLM question only when it agrees with the writer on both heads:
   4,161 of 7,275 questions are clean. Agreement by class shows where the writer drifts: comparison 0.93, result/simple 0.84,
   method/simple 0.56, factual/simple 0.52, survey/complex 0.22, method/complex 0.17, factual/complex 0.14 (multi-part questions
   of one intent are often a comparison to the blind pass).
2. **Iteration 1** (2,783 train): targets met from epoch 1; best epoch 5 composite 0.9867. Only real error: "Which of TinyBERT or
   MobileBERT keeps more of BERT's GLUE performance?" was called simple factual. The templates repeat their wording and inflate
   the validation score, so from iteration 2 the validation score is also reported on the LLM-written part (val_llm) and the
   composite uses it; dev C was written to have one clean held-out set.
3. **Iteration 2** (3,667 train): new comparison and multi-part templates ("which of X or Y", "list the languages ... and the
   score of ... on each"), 2,336 targeted LLM questions for the weak classes. Best epoch 3, composite 0.983, dev A 1.000; missed
   the val_clean intent bar by 0.001 (0.9689 vs 0.97). Dev C 48/48.
4. **Iteration 3** (5,157 train = iteration 2 + 1,490 perturbed copies: lower case, dropped question mark, a swapped-letter typo, a
   chatty prefix): best epoch 5, composite 0.9825 (no gain over iteration 2, 0.0005 lower = noise) but **all six targets met**, and
   clearly better on typed-like-a-user dev questions (dev A perturbed x3 100 % vs 97.6 %). Installed.
Stop rule: targets met in an iteration >= 2 -> campaign over. Runs stopped by the monitor agents (Haiku, one per iteration 2 and 3,
polling `metrics.jsonl` every ~10 s, creating `STOP` on "three evaluations without gain" / two consecutive target hits). A run takes
1.3-2.3 min, so the agents matter little for speed; iteration 1 was shorter than the agent's poll and ran without one.

## Effect on the pipeline
- A comparison question now comes out complex, so Stage 2's guardrail rewrites and decomposes it ("How does XLM-R compare with
  mBERT on XNLI?": intent comparison / complex; the zero-shot LLM used to call some of these simple).
- Stage 1 cost: SciBERT on the GPU 5.7 ms per question (measured); the 12B call for the conditions stays (~1.6 s).
- Smoke test (5 questions, warm): median 8.3 s per question end to end (7.7-9.3 s), first question 33 s (model warm-up).

## Reproduce
```
python scripts/make_template_questions.py --out RUN/template.jsonl --per-class 200 --seed 31
python scripts/make_query_training_data.py --out RUN/llm_generated.jsonl --kinds ... --styles ... --seed 22 --no-seeds
python scripts/prepare_query_data.py --inputs A.jsonl B.jsonl T.jsonl --out-dir RUN/data --relabel --augment 0.5
python scripts/train_query_classifier.py --train RUN/data/train.jsonl --val RUN/data/val_clean.jsonl --run-dir RUN --iteration N
python scripts/install_query_classifier.py RUN
python scripts/eval_stage1.py data/index/query_classifier --errors-for A,B,C [--perturb 5]
```
Run folders: `data/index/query_runs/iter1..iter3` (data, metrics.jsonl, status.json, errors.json, logs).

## Limits (say them in the paper)
- Silver labels; dev sets are assistant-written; the classifier was never scored on human-written questions until labelling job B.
- The two-class complexity has no "middle" level; a question with one condition but several numbers can go either way.
- The blind relabel uses the same 12B model family that wrote the questions: correlated errors are possible.
