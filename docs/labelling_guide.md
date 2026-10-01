# Labelling guide (1 October 2026, due 3 October)

The labels are the **answer key for the evaluation**. Nothing is trained on them. They tell us how good the finished system is:
how often the extractor is right (job A), whether the system warns when the papers do not cover a question (job B), and whether it
tells a real conflict between two papers from a difference that conditions explain (job C).

## Who does what
| Job | What you do | Who | Time | File |
|---|---|---|---|---|
| **A** profile check | check what the system extracted from a table row (verdict per row) | Adarsh, Shashank | 170 rows each, ~1 min per row (about 3 h) | `A_profile_check_<name>.xlsx` |
| **B** questions | write 7-8 test questions and say what the system should do | all four | ~5 min per question | `B_questions_<name>.xlsx` |
| **C** result pairs | decide for 50 pairs of conflicting numbers: real conflict or explained? | Aditya and Tarun, **each all 50, alone**; a third person settles disagreements | ~5 min per pair (about 4 h) | `C_result_pairs_<name>.xlsx` |

Every workbook starts with a **READ ME** sheet: what to do, definitions, an example, a live "rows done" counter. The data sheet has
**yellow** cells (yours), **grey** cells (given, locked) and turns a verdict cell **green** when it is filled in.

## The rules
1. Change only yellow cells. The sheets are protected without a password (to stop accidents); do not unprotect them.
2. Judge **from the papers**. Every row has an **open** link to the exact arXiv page. Do not look at the system's answers first, and
   never paste profiles, answers or sheet contents into a chat tool.
3. Gemini may help you *understand* a table layout (paste the PDF text only) or *draft* questions. It is not a labeller: the verdict
   is yours (it is the same model family as the system's LLM, so its mistakes would match the system's).
4. Job C: Aditya and Tarun do **not** discuss until both are finished (we report Cohen's kappa between you two; the target is 0.6).
5. The last 20 rows of each job-A sheet are shared with the other labeller to measure agreement. Do them like the others.
6. Save often. When done, save a copy with `_DONE` added to the name and send it back.

## What the numbers will become in the paper
* Job A: field-level precision / recall / F1 of the Condition Profile Extractor (per field, easy vs hard papers), profile-level precision.
* Job B: intent and complexity accuracy of the Stage 1 classifier on questions nobody trained on; scope-warning precision / recall of Stage 6
  (does it warn when it should and stay quiet when it should?).
* Job C: Cohen's kappa of the two labellers; accuracy, per-class F1 and confusion of Stage 7's GENUINE / EXPLAINED / NOT COMPARABLE.

## Files and keys (assistant's side)
* Built by `python scripts/make_label_sheets.py` into `data/labelling/` (git-ignored). `data/labelling/private/` holds the answer keys
  (the system's verdict per pair, who got which profile): do not send those around.
* Sheets are built from the **current profile store**. If a paper is re-ingested, rebuild the sheets first.
* The corpus-map sheets in the job-B workbooks (Papers, Datasets, Dataset x Language, Models, Model x Dataset) are built from the
  extractor's output. They help you find out whether a condition is covered, but they can be wrong: confirm in the paper.

## Scoring (after the files come back)
```
python scripts/score_labels.py status data/labelling/done/*.xlsx                        # who is how far
python scripts/score_labels.py A  data/labelling/done/A_*_DONE.xlsx                      # -> eval/labels/job_A_scores.json
python scripts/score_labels.py B  data/labelling/done/B_*_DONE.xlsx                      # -> eval/labels/questions_gold.jsonl
python scripts/eval_questions.py eval/labels/questions_gold.jsonl                       # runs the system on those questions, scores Stage 1 + 6
python scripts/score_labels.py C --first C_..Aditya_DONE.xlsx --second C_..Tarun_DONE.xlsx \
        --make-adjudication data/labelling/C_adjudication.xlsx                          # kappa + the pairs the third person settles
python scripts/score_labels.py C --first ... --second ... --third C_adjudication_DONE.xlsx   # final gold + the system's accuracy
```

### Definitions used in the scores
* **Job A, field f**, over profiles judged a real result (ALL_OK or SOME_WRONG): TP = filled and right; FP = filled and wrong (WRONG);
  FN = the paper states a value the profile does not have right (WRONG or MISSING). Precision = TP / (TP + FP), recall = TP / (TP + FN),
  F1 = harmonic mean. An empty cell that the paper does not state counts for nothing. NOT_A_RESULT rows lower the *profile-level*
  precision (real results / judged rows) and are left out of the field scores; CANNOT_CHECK rows are left out of everything.
* **Job B**: a question counts as "warning expected" when the labeller wrote YES. The system "warns" when Stage 6 returns a scope
  warning. Precision = warnings that were expected / warnings given; recall = expected warnings that were given.
* **Job C**: the gold verdict is the two labellers' verdict when they agree, otherwise the third person's. For EXPLAINED, the gold
  conditions are those that every labeller for that verdict ticked.
