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
  accuracy >= 0.90. Dev B is reported at the end (it should be within ~5 points of dev A).
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
- iteration 1: pending (data generation was running at 2026-09-30 16:45)
