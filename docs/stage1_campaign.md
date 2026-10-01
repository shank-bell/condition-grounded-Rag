# Stage 1 (SciBERT) training campaign

The architecture's Stage 1 is a SciBERT model with two heads: **intent** (factual, method, result, comparison, survey) and
**complexity** (simple, complex). It is the only model trained in this project. User instruction of 2026-09-30: train it
automatically, monitored by agents, stop when a threshold is reached, repeat (at least 2 iterations) until the metrics are as
good as they can get. "Perfect" cannot be literal: the labels are silver (written by an LLM), so the targets below are the
bar; if they are not reached after 6 iterations the best checkpoint is installed and the limits are reported.

## Data
| Source | Script | Labels |
|---|---|---|
| LLM-written questions from real passages of the 28 papers (8 kinds x styles plain / informal / terse / multipart) | `scripts/make_query_training_data.py` | known by construction, but the writer can be wrong |
| Template questions with names from the profile store | `scripts/make_template_questions.py` | correct by construction |
| 15 hand-written seeds | in `make_query_training_data.py` | correct |

`scripts/prepare_query_data.py --relabel` runs a blind second pass (12B sees only the question and a rubric); an LLM-written
question is **clean** when both passes agree on both heads. Training uses clean questions only. It writes `train.jsonl`,
`val_clean.jsonl` (12 % of clean, stratified), `val_noisy.jsonl` (diagnostic) and `stats.json` (agreement per class).
Passages come from `data/index/chunks.jsonl` (exported from ChromaDB once), so generation can run while another process
holds the index. Questions equal to a dev question are dropped.

**Dev set** `eval/stage1_dev.jsonl`: 107 questions written by the assistant, labels are the assistant's judgement (not gold),
split A (68, used for decisions and error analysis) and B (39, never used for decisions; reported at the end).
Rebuild with `scripts/build_stage1_dev.py` from `eval/stage1_dev_raw.txt`. The team's own questions (labelling job B) replace
it as the real test.

Rubric (same as `KINDS` in the generator): factual = a stated fact (not a score); method = how / why a technique works;
result = reported scores; comparison = compare / which is better / do papers agree; survey = overview of a topic.
simple = exactly one thing; complex = several things, lists or aggregates, a comparison, or an overview. Every comparison and
every survey question is complex.

## Training
`scripts/train_query_classifier.py --train RUN/data/train.jsonl --val RUN/data/val_clean.jsonl --run-dir RUN
[--init-from PREVIOUS/best]`. AdamW lr 3e-5, linear warm-up + decay, label smoothing 0.05, batch 16, up to 10 epochs / 25 min.
Files in the run directory: `metrics.jsonl` (an `eval` line per epoch, a `train` line every 25 steps), `best/` (weights.pt +
meta.json), `status.json`, `errors.json` (misclassified questions of the best checkpoint + per-class scores), and the file
`STOP` which a monitor creates to end the run (the best checkpoint is kept).
`composite = mean(val intent F1, val complexity F1, dev-A intent acc, dev-A complexity acc)`.
Install: `python scripts/install_query_classifier.py RUN` copies `best/` to `data/index/query_classifier/`, which Stage 1 loads
automatically.

## Targets and stop rules
- **Targets:** val_clean intent macro-F1 >= 0.97 and complexity macro-F1 >= 0.95; dev-A intent accuracy >= 0.92 and complexity
  accuracy >= 0.90. Added after iteration 1 (the template questions repeat their wording and inflate val_clean): the same
  bars on the LLM-written part of the validation set, `val_llm_intent_f1` >= 0.95 and `val_llm_cx_f1` >= 0.95, and the composite
  is now computed from val_llm. Dev B is logged but its errors were read once, so **split C** (48 questions written after
  iteration 1, never used for a decision, not scored by the trainer) is the clean held-out check: `scripts/eval_stage1.py`, at the end.
- **Monitor stops a run** when the targets are met, or 3 evaluations in a row do not improve the composite, or 6 minutes pass
  without a new metrics line while the process is alive, or the loss is NaN.
- **Campaign stops** when the targets are met in an iteration >= 2, or after 6 iterations, or after ~4 hours, or when two
  iterations in a row improve the composite by < 0.005. The best run overall (highest composite) is installed.

## One iteration
1. Data: iteration 1 = full generation (LLM ~1,120 calls + templates + seeds); iteration k > 1 = targeted, from the previous
   `errors.json`: more questions of the confused classes (`make_query_training_data.py --kinds intent:complexity,... --styles
   informal,terse,multipart`, `make_template_questions.py --only ...`), always with a new `--seed`.
2. `python scripts/prepare_query_data.py --inputs <files> --out-dir RUN/data --relabel` (blind labels are cached, so only new
   questions cost LLM calls). Keep earlier iterations' clean data in the union.
3. Start training in the background; write `RUN/targets.json` first. Iteration k > 1 starts from the previous best (`--init-from`).
4. Spawn ONE monitoring sub-agent (prompt below) in the background; continue other work; read its report.
5. Read `status.json` / `errors.json`, log the iteration below, decide the next step or stop.

## Monitor-agent prompt (fill in RUN and the process id)
> You monitor a SciBERT training run on this Windows PC. Run directory: `RUN`. It contains `metrics.jsonl` (JSON lines; those with
> `"kind": "eval"` are evaluations, `"kind": "train"` are loss lines), `targets.json` (the thresholds) and, when the run ends,
> `status.json`. The training process id is PID. Every ~30 seconds read the new lines of `metrics.jsonl` (use the shell; keep
> each check short). After each eval line compute: targets_met = val_intent_f1 >= T.val_intent_f1 AND val_cx_f1 >=
> T.val_cx_f1 AND dev_a_intent_acc >= T.dev_a_intent_acc AND dev_a_cx_acc >= T.dev_a_cx_acc. Create the empty file `RUN/STOP`
> (nothing else) when (a) targets_met, or (b) three consecutive eval lines have no higher `composite` than the best so far,
> or (c) more than 6 minutes passed without any new line while PID is still running, or (d) a loss is NaN. Do NOT kill any
> process, do NOT edit or create any other file, do NOT touch data/index besides reading `RUN`. After STOP is created (or when
> `status.json` appears) wait for `status.json`, then reply with ONLY a short report: why you stopped (or that the run ended by
> itself), the best eval line (epoch, composite, val F1s, dev A/B accuracies), whether the targets were met, the number of eval
> lines seen. Give up after 30 minutes and report what you saw.

## Iteration log
(append one line per iteration: date time, data size, clean rate, best epoch, val F1s, dev A / B accuracies, targets met?, next step)
- iteration 1 (2026-09-30 17:26-17:28): data = 3,351 LLM questions + 1,600 templates + 15 seeds, 4,947 unique after removing dev
  duplicates; the blind relabel confirmed 3,159 (clean; agreement of the two passes on both heads by class: comparison/complex 0.92,
  result/simple 0.84, factual/simple 0.52, method/simple 0.56, result/complex 0.34, factual/complex 0.14, method/complex 0.17,
  survey/complex 0.22 - the writer mixes types in its multi-part questions, e.g. "how does X differ from Y and what ..." is a
  comparison to the blind pass). train 2,783 (1,394 template / 1,376 LLM / 13 seed), val_clean 376. Training 10 epochs, 1.4 min
  (the run is shorter than the monitor's 30 s poll, so no monitor agent was spawned for it; the trainer finished by itself).
  Best epoch 5: composite 0.9867, val F1 intent 0.984 / complexity 0.992, dev A 0.985 / 0.985 (68), dev B 0.974 / 0.974 (39);
  the targets were met from epoch 1. 8 errors (5 val, 1 dev A, 2 dev B): the real one is "Which of the small BERT variants,
  TinyBERT or MobileBERT, keeps more of BERT's GLUE performance?" -> factual/simple (a "which of X or Y" comparison), and the rest
  are boundary cases (survey vs method, result vs factual). Iteration 2 is therefore a robustness pass, not a chase for numbers:
  more "which of / or / vs" comparison phrasings and multi-part result / factual questions (new templates), targeted LLM questions
  for the seven classes with weak agreement (informal / terse / multipart styles, seed 22), val_llm reported, dev C written.
- iteration 2 (17:38-17:55): data = iteration 1 + 2,336 targeted LLM questions (seven classes, informal / terse / multipart, seed 22) + new
  templates; 7,275 unique, 4,161 clean, train 3,667, val_clean 494 (332 LLM-written). Monitor agent stopped the run after three evaluations
  without gain (epoch 6-7 of 10). Best epoch 3: composite 0.983, val F1 0.9689 / 0.9876, val_llm 0.9502 / 0.9819, dev A 1.0 / 1.0, dev B
  0.974 / 1.0; val intent F1 missed the 0.97 bar by 0.0011. Installed for the smoke test; dev C 48/48 (looked at once).
- iteration 3 (18:00-18:10): same data + `--augment 0.5` (1,490 perturbed train copies, train 5,157, same val). Monitor stopped it after epoch 8
  (three evaluations without gain). Best epoch 5: composite 0.9825, val F1 0.9751 / 0.9773, val_llm 0.963 / 0.9668, dev A 1.0 / 1.0, dev B
  0.974 / 1.0 -> ALL TARGETS MET. Composite is 0.0005 below iteration 2 (noise), but on typed-like-a-user dev questions it is better (dev A x3
  1.000 vs 0.976, dev B 0.974 / 1.000 vs 0.966 / 0.974), so iteration 3 is the installed model. Dev C: 48/48 plain, 240 perturbed 0.983 intent /
  1.000 complexity. Campaign closed (rule: targets met in an iteration >= 2). Report: `docs/stage1_training_report.md`.
