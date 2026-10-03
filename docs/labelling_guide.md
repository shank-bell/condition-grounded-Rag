# Labelling guide (1 October 2026, due 3 October)

The labels are the **answer key for the evaluation**. Nothing is trained on them. They tell us how good the finished system is:
how often the extractor is right (job A), whether the system warns when the papers do not cover a question (job B), and whether it
tells a real conflict between two papers from a difference that conditions explain (job C).

## Why we are labelling (the detailed version)
**The system makes three claims that nobody has checked against the truth yet:** (1) it reads numbers out of paper tables correctly, (2) it
warns you when the papers do not cover what you asked, (3) it tells a real disagreement between two papers from one that different
experimental conditions explain. Labelling is how humans create the truth to test those claims against: an **answer key**, like a marking
scheme; the system is the student. Nothing is trained on it.

**Why we need an answer key at all.** Everything the system "knows" comes from an AI reading tables, and that AI can be wrong. The tests the
assistant ran itself use the system's own data as the reference, so they can only detect "the system disagrees with itself": if the AI misread
a table and a result never entered the store, such a test cannot notice. Only a person reading the actual paper can say what is true.
Without that the paper could only say "it seems to work".

**Why humans, not Gemini or our own model.** The answer model is Gemma and Gemini is the same kind of model, so their mistakes would match and
hide each other. The project's design document also requires human agreement (Cohen's kappa). Gemini may help someone understand a table
layout; the verdict is always a person's.

**Job A - is what the system extracted correct?** A "profile" is one row per reported number, e.g. `BERT-base | MNLI-m | accuracy | 84.4 |
English | fine-tuned`, up to nine fields. The two novel parts read these profiles, so a wrong profile means a wrong warning or conflict verdict.
(The assistant's own spot check of 84 profiles found ~58 % fully right: a tiny sample by the wrong person.) Result: precision, recall and F1
per field, split into 6 easy and 4 hard papers, plus where the AI fails (e.g. a dataset recorded as "Dev Set SST-2"). The 20 shared rows
show whether two people judge the same way.

**Job B - does the system warn at the right time?** Everyone writes 7-8 questions as a student would, and says what the right behaviour is:
10 fully covered ("How well do models perform on Kannada NLI?" is answerable: IndicXNLI), 10 with one condition missing ("XLM-R on XNLI for
Kannada": XNLI has 15 languages, none is Kannada, so the system must warn), 10 partly covered ("Compare mBERT and GPT-4 on XNLI": GPT-4 is in
no paper). Two ways to fail: a false warning is annoying; a missed warning is worse (someone trusts an answer that does not apply to them).
The questions must be written FROM THE PAPERS: if they came from the system's store we would never find a result the extractor missed. Result:
warning precision / recall, intent and complexity accuracy on real questions nobody trained on, how often the right missing condition is named.
If a question names several models or languages, write them all in the cell separated by commas ("mBERT, XLM-R, GPT-4").

**Job C - real conflict or explained difference?** 50 pairs where two papers report different numbers for "the same" thing. EXPLAINED (the
experiments differ: human EM on SQuAD 82.3 vs 86.9 is v1.1 vs v2.0; tick which condition differs), GENUINE (same conditions, still different),
NOT COMPARABLE (cannot tell). The judgement is partly subjective, so two people label independently and we measure how often they agree beyond
chance (kappa, target 0.6 or higher); a third person settles the disagreements; the system's own verdict is hidden from everybody. Result:
kappa, and the system's accuracy, per-class F1 and confusion on genuine vs explained.

**Why we do not train on these labels:** only the intent / complexity classifier is trained (on AI-written questions); 30 questions and 50 pairs
are far too few, and training on the test would make our own results meaningless.

**Why the rules:** judge from the papers (otherwise the system's mistakes are copied into the answer key); Job C labellers do not discuss
until both are done (otherwise kappa means nothing); no system answers are shown; the assistant scores FIRST and freezes the numbers, later fixes
are reported separately (so we cannot tune the system to its own test).

**What happens next:** files back by 3 Oct; evaluation and ablations (escalation, joint coverage on / off, on the team's real questions) on
4-5 Oct; numbers into the paper; code freeze on day 10, then writing. Without these labels the system would work but we could not prove it.

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
