# Evaluation against AI-annotated answer keys (draft of 8 October 2026)

**Read this first.** No human ever labelled anything for this project: the teammates' label files never arrived. With the project lead's
approval (7 October) the answer keys were written by an AI assistant (Claude) that read the original PDFs. That means:

- **one labeller**, so no agreement score (Cohen's kappa) can be given;
- the labeller **also helped to build the system** it is judging, so it is not independent of the builder;
- the project design asked for human labels (for Stage 7: two humans, kappa of at least 0.6). That did not happen, and the paper has to say so.

Every number in this document is therefore "**against an AI-annotated key**". The human route stays open: real label files go through
the same scoring scripts without any change (section 8).

All numbers come from one PC (RTX PRO 4000, 24 GB; answer model Gemma 12B, the two agents Gemma E2B). The architecture document says
RTX 4050 6 GB, so the paper needs a footnote on the machine.

## The results in one page

"Frozen" = code of commit `ca007a6`, run before any failure was looked at (the headline). "After fixes" = the same code plus three general bug fixes made after reading the failures (not held-out; section 6).

| Claim | What the data say | How sure |
|---|---|---|
| 1. Extraction | 98 % of 170 sampled profiles are real results and the numbers in them are right 99 % of the time. Names are weaker (F1: task 79, model 84, dataset 86). Field F1 90.5 % lenient / 85.8 % strict. About half of the profiles are completely right (53 % lenient, 41 % strict); hard papers are worse (44 % / 33 %). | 170 profiles, CI about +-7 points. No baseline. |
| 2. Stage 6 scope warning | Warns on 19 of 19 questions that need it, falsely on 1 of 11 after the fixes (3 of 11 frozen); F1 97.4 after fixes, 92.7 frozen. A simple LLM judge scores F1 91.4: a tie on this sample (question by question p = 0.63), but the judge answers only yes / no and misses 3 of 19, while Stage 6 names the missing condition (17 of 19 right). Plain RAG and abstain-on-weak-retrieval never warn. | 30 questions: one question moves F1 by about 3 points. |
| 2b. Which parts matter | Only the joint-coverage check: without it F1 falls to 48. The two LLM agents (as fixed rules), escalation, retrieval cards and profile-guided retrieval change no decision on 30 questions. | "Not detected at this size", not "useless". |
| 3. Stage 7 | Calls 7 of 50 pairs GENUINE where the key has none, against 13 for the Gemma baseline and 48 for plain NLI. But accuracy is 62 % (Gemma baseline 60 %, "always explained" 82 %), so it is not more accurate than the baseline. | 50 pairs, **no genuine pair in the key**: finding a real contradiction is untested. |
| 4. Answer quality | 5 of 9 correct with and without the profile stages, 0 of 9 for the language model alone: parity, not an improvement. | 9 questions; p = 0.0625 for 5-0. |

**Honest reading for the paper.** The condition store works as a scope checker, and the one idea that carries the result is the joint-coverage check. The data do not show that the LLM agents or Stage 7 beat simple alternatives, and claims should be worded to that.
Everything is measured against keys made by one AI labeller; swapping in human labels is a re-run of the same scripts (section 8).

## 1. What was tested

The architecture names four claims. Each is scored against its own key.

| # | Claim | Measured as | Key (size) | Compared with |
|---|---|---|---|---|
| 1 | The Condition Profile Extractor stores correct profiles | per-field precision / recall / F1 and the share of fully correct profiles | A: 170 random stored profiles from 10 papers | nothing (the planned MetaLead-style baseline was not built) |
| 2 | Stage 6 warns when, and only when, a named condition is not covered | precision / recall / F1 of "warning given", right condition named | B: 30 questions, 19 should warn, 11 should not | plain RAG, abstain-on-weak-retrieval, an LLM judge (Sufficient-Context style, Gemma 12B), the 30 September version of Stage 6, and six ablations |
| 3 | Stage 7 tells an explained difference from a genuine contradiction | accuracy, macro-F1, false-conflict count, condition named | C: 50 result pairs | plain NLI, Gemma 12B with the conflict taxonomy |
| 4 | The profile parts do not hurt answer correctness | an answer is correct when it states the key's numbers | B: the 9 answerable questions that have numeric facts | pipeline without the profile stages (C / 6 / 7), LLM alone |

Stage 1 (question understanding) is scored on the same 30 questions: intent, complexity and the conditions it extracts.

## 2. How the keys were made

**Job A, profile check (170 rows).** A random sample of 170 stored profiles from 10 papers: 6 "easy" papers (BERT, SQuAD 2.0, XNLI, XLM-R, GLUE, MuRIL;
99 profiles) and 4 "hard" ones (T5, GPT-3, LLaMA, Llama 2; 71 profiles). For each profile the labeller looked at the table row on the PDF page and gave a
verdict: ALL_OK, SOME_WRONG (with the wrong or missing fields), or NOT_A_RESULT. A cell that is true but too vague (for example setting "fine-tuned" where
the block says "translate-train-all") is marked `imprecise`; it is counted as right in the **lenient** reading and as wrong in the **strict** reading, and both are reported.
A wrong cell counts as one false positive and one false negative; a missing cell as one false negative. Rows 21-170 were marked without any other AI's answer for them (none ever existed);
rows 1-20 were marked after seeing a Gemini draft of those 20 rows (Gemini made one clear error and was inconsistent, so that route was dropped after 20 rows).

**Job B, questions (30).** 11 questions are fully covered by the 28 papers, 9 name one condition the papers do not cover, 10 are partly covered; so 19 should
get a scope warning. Each question has an intent, a complexity, the conditions it names, the condition that is missing (if any) and, for the 9 answerable questions with
numbers, the facts the answer must contain. The facts were read from the PDF pages. **Every claim "paper X does not report Y" was checked with a raw-text search over
all 28 PDFs.** The labels were fixed before the system saw any question. Seven of the question texts were adapted from a draft sheet of unknown origin
whose labels were wrong; the questions were kept and the labels redone. The questions were written by someone who knew which failure Stage 6 is built for (a model, a dataset
and a language that exist separately but never together), so the set favours Stage 6's design to some extent.

**Job C, result pairs (50).** Two stored results from two papers that differ numerically. The labeller judged from the papers (not from the system) whether the
difference is EXPLAINED by a different condition (and which one), GENUINE (same conditions, still different) or NOT COMPARABLE. Result: 41 explained, **0 genuine**, 9 not
comparable; 7 pairs are marked "the stored profile is wrong" and 7 as low confidence (for example the SQuAD human-performance numbers of 2016 against the leaderboard
numbers could also be read as a genuine difference).

**Rules that keep the numbers honest.**
- Labels are an answer key, never training data. Nothing was trained on them (the only trained model is the Stage 1 SciBERT, on LLM-written questions).
- The key comes first, the system's verdicts after (one exception, below). The Stage 7 baselines were frozen at commit `5c9fc56` before any label existed.
- The headline numbers are from the code frozen at commit `ca007a6`. Fixes made afterwards are reported separately and labelled "after error analysis, not held-out" (section 6).
- Every result file carries a freeze record (commit, modified files, time) written by `src/cgrag/evaluation/freeze.py`.

## 3. Corrections log (everything that went wrong, in order)

1. **4 Oct, job A: 28 of 49 page views failed silently.** The image tool returned "[media removed: request limit]" instead of a picture and I did not notice, so those rows were
   first judged from the table text in the sheet (the same text the extractor read) plus text searches. On 7 Oct all 28 pages were re-read as pictures; all 170 verdicts held.
2. **Job B, two label errors found after the first system run** (both kept in the provenance file; key v1 kept next to v2): B-AI-17 "DistilBERT F1 on SQuAD 2.0" is covered
   (MobileBERT Table 5 reports DistilBERT-6L EM 66.0 / F1 69.5) - the system was right and the label wrong; and the mT5-Large Swahili fact is 65.7 / 45.3 (I had read the Russian column).
   The first absence check had only searched one paper; since then every absence is searched in all 28 PDFs.
3. **Job C, first label set redone** after re-reading the real tables (nine pairs changed, for example a DistilBERT average of accuracy and F1 compared with an accuracy). This happened before any system verdict was looked at.
4. **Job C, one slip:** the system's verdicts for pairs P01-P03 were printed once by accident after those three labels had been decided. The labels were not changed.
5. **Teammates' draft sheets** found in the Downloads folder (origin unknown): a questions sheet (wrong labels, see above) and a pre-filled job A sheet with invalid verdict names that flagged only 17 of the 82 problem rows found here. Neither was used as a key.
6. **Fixes after error analysis** (section 6): made after seeing the failures, so those numbers are not held-out.
7. **The clean re-run was started twice (8 Oct, 00:21 and 00:30).** The second start was not made by the assistant and its cause is unknown. For six minutes two pipeline processes shared the GPU. No decision changed, but the timings of
   two tests were inflated and one Stage 1 reading differed (section 5). Both tests were run again alone from the same commit (01:02-01:16) and the tables use those repeats.

## 4. Results

All numbers are against the AI-annotated keys. "95 % CI" is a Wilson interval and shows how little 30 questions or 50 pairs can say.
"Frozen" means the code of commit `ca007a6`, run before any failure was looked at. "Clean re-run" means the code of commit `4d06f84` (the same code plus the three fixes of section 6), run again
from one commit on 8 October; those numbers are "after error analysis, not held-out".

### 4.1 Claim 1: are the stored profiles right? (key A, 170 profiles)

**Verdict: the numbers are right, the names are the weak part, and about half of the profiles are completely right.**
167 of the 170 sampled profiles are real results (98.2 %). Of those, 88 are completely right (52.7 %, 95 % CI 45-60) when a true-but-vague cell counts as right (lenient),
and 68 (40.7 %, CI 34-48) when it counts as wrong (strict). On the 6 easy papers 58.8 % / 46.4 % are completely right, on the 4 hard papers (T5, GPT-3, LLaMA, Llama 2) 44.3 % / 32.9 %.

| Field | precision % | recall % | F1 % lenient | F1 % strict | F1 % easy papers (lenient) | F1 % hard papers (lenient) |
|---|---|---|---|---|---|---|
| value | 99.4 | 99.4 | 99.4 | 99.4 | 100.0 | 98.6 |
| language | 97.8 | 97.8 | 97.8 | 97.8 | 96.2 | 100.0 |
| metric | 97.0 | 97.0 | 97.0 | 96.4 | 99.0 | 94.3 |
| setting | 93.3 | 93.3 | 93.3 | 72.1 | 93.7 | 92.9 |
| model_size | 89.8 | 89.8 | 89.8 | 89.8 | 90.9 | 89.2 |
| dataset | 86.5 | 84.4 | 85.5 | 81.8 | 88.4 | 81.4 |
| model | 84.4 | 84.4 | 84.4 | 76.6 | 92.8 | 72.9 |
| task | 78.7 | 78.7 | 78.7 | 78.0 | 84.2 | 71.0 |
| dataset_version | 100.0 | 58.8 | 74.1 | 74.1 | 83.3 | 0.0 |
| **all fields (micro)** | 90.9 | 90.1 | **90.5** | **85.8** | 93.2 | 86.9 |

How to read it: the 9 fields are scored on every real result; a wrong cell is one false positive and one false negative. Almost every error is in a *name*, not in a number:
a data set name in the task field; a cut or garbled model name ("lama 1", "Fln", "Wikipedia"); "Dev Set X" or a language pair stored as the data set; two stacked table blocks mixed
up (GPT-3 SuperGLUE, IndicCOPA); a side-by-side table with the value under the wrong model (LLaMA MMLU); a category such as ToxiGen, BOLD or WinoGender not recorded.
`dataset_version` is missing in 7 of the 17 cases that have one, and it is stored for only 4.4 % of all profiles. The setting field is where lenient and strict differ most (93.3 vs 72.1),
because many settings are true but vague. There is **no baseline** for this claim: the MetaLead-style extractor named in the plan was not built.

### 4.2 Question understanding (Stage 1, the same 30 questions)

**Verdict: intent and complexity are read well; the named conditions are mostly right after the fixes (precision 95.7 %, recall 89.1 %), but sizes, settings and the task field are weak.**

| Measure | frozen (`ca007a6`) | clean re-run (after the fixes) |
|---|---|---|
| intent accuracy | 93.3 % (28 / 30) | 93.3 % |
| intent macro-F1 | 96.9 % | 96.9 % |
| complexity accuracy | 100 % (30 / 30) | 100 % |
| conditions: precision | 86.5 % | 95.7 % |
| conditions: recall | 82.2 % | 89.1 % |
| conditions: F1 | 84.3 % | 92.3 % |
| median seconds per question (Stages 1-6) | 3.9 | 4.4 |

Per condition (clean re-run): model 42 of 42, dataset 26 of 27, language 16 of 17, dataset version 3 of 3. Setting 2 of 5 (the two misses are "5-shot"). Model size 0 of 6: Stage 1 keeps the size inside
the model name ("LLaMA 65B", "Mistral 7B"), and since the fix of section 6 Stage 6 reads the size from there, so this is a bookkeeping gap and not a lost condition. The task field is unreliable:
the key names a task for one question only, and Stage 1 filled the field for five, two of them wrongly ("zero-shot" and "MLQA", which is a data set).

### 4.3 Claim 2: does Stage 6 warn when, and only when, a condition is not covered? (key B, 30 questions: 19 should warn, 11 should not)

**Verdict: Stage 6 warned on all 19 questions that needed it and falsely on 1 of 11, and it names the missing condition correctly in 17 of 19. A simple LLM judge scores about the same, and 30 questions cannot separate them (sign test p = 0.63).**

The systems, in plain words: *plain RAG* never checks scope. *Abstain on weak retrieval* warns only when the retrieval score is low (it never fired). The *LLM judge* (the "sufficient context" idea, Gemma 12B)
reads the question and the retrieved passages and says whether they are enough to answer; "not enough" counts as a warning. The *30 September Stage 6* checks each condition on its own. *Stage 6 + joint coverage* adds the check that the named model, data set and
language are recorded **together** by one result; *shipped* also has the second opinion of the bigger model (escalation).

| System | warned | TP | FP | FN | precision % (95 % CI) | recall % (95 % CI) | F1 % clean | F1 % frozen | seconds / question |
|---|---|---|---|---|---|---|---|---|---|
| Plain RAG (never warns) | 0 | 0 | 0 | 19 | - | 0.0 (0-17) | 0.0 | 0.0 | 4.1 |
| Abstain on weak retrieval | 0 | 0 | 0 | 19 | - | 0.0 (0-17) | 0.0 | 0.0 | 4.1 |
| LLM judge (Gemma 12B) | 16 | 16 | 0 | 3 | 100.0 (81-100) | 84.2 (62-94) | 91.4 | 91.4 | 6.2 |
| Stage 6, version of 30 September | 9 | 9 | 0 | 10 | 100.0 (70-100) | 47.4 (27-68) | 64.3 | 64.5 | 4.6 |
| Stage 6 + joint coverage | 20 | 19 | 1 | 0 | 95.0 (76-99) | 100.0 (83-100) | 97.4 | 92.7 | 4.6 |
| **Stage 6 shipped** | 20 | 19 | 1 | 0 | 95.0 (76-99) | 100.0 (83-100) | **97.4** | **92.7** | 5.2 |

(The frozen run of the shipped system had 3 false alarms, precision 86.4 %; the fixes of section 6 removed two of them. The seconds are from the undisturbed repeat; see section 5.)

Question by question, Stage 6 is right on 29 of 30 and the judge on 27. They disagree on four questions: Stage 6 is right on three of them, the judge on one (p = 0.63, no difference). Against the 30 September version it is right where the
older one is wrong on 10 questions and the other way round on 1 (p = 0.012); plain RAG is right on only 11, the questions that need no warning (p < 0.001). In the frozen run, before the fixes, the picture was a tie: F1 92.7 against 91.4.
The two systems fail differently: the judge never raises a false alarm but misses 3 of 19 and answers only yes or no; Stage 6 misses none, raises one false alarm, and tells the user *which* condition is missing and what the papers do record.
Stage 6 costs about one second per question on top of plain RAG (5.2 against 4.1 s); the judge costs two (6.2 s). The one false alarm left (B-AI-07, IndicCOPA, Tamil) is an extraction failure: Table 17 of 2212.05409 was stored with the wrong header (section 7).

### 4.4 Which parts matter? (ablation of the scope warning, same 30 questions)

**Verdict: only the joint-coverage check matters. Everything else gives identical decisions on all 30 questions.**

| Variant | frozen TP / FP / FN | F1 % frozen | after fixes TP / FP / FN | precision / recall % | F1 % after fixes |
|---|---|---|---|---|---|
| Shipped system | 19 / 3 / 0 | 92.7 | 19 / 1 / 0 | 95.0 / 100.0 | 97.4 |
| No escalation | 19 / 3 / 0 | 92.7 | 19 / 1 / 0 | 95.0 / 100.0 | 97.4 |
| **No joint-coverage check** | 8 / 2 / 11 | 55.2 | 6 / 0 / 13 | 100.0 / 31.6 | **48.0** |
| Orchestrator (Stage 2) as fixed rules, no LLM agent | 19 / 3 / 0 | 92.7 | 19 / 1 / 0 | 95.0 / 100.0 | 97.4 |
| Applicability (Stage 6) as fixed rules, no LLM agent | 19 / 3 / 0 | 92.7 | 19 / 1 / 0 | 95.0 / 100.0 | 97.4 |
| No retrieval cards | 19 / 3 / 0 | 92.7 | 19 / 1 / 0 | 95.0 / 100.0 | 97.4 |
| No profile-guided retrieval | 19 / 3 / 0 | 92.7 | 19 / 1 / 0 | 95.0 / 100.0 | 97.4 |

Why the joint check matters: in "XLM-R on XNLI for Kannada" every part exists in the papers (XLM-R, XNLI, Kannada), and only the *combination* is missing; a check that looks at each part alone says "covered".
Without the joint check Stage 6 warns on only 6 of the 19 questions. **"No effect" here means "not detected on 30 questions"**: the two LLM agents decide exactly like fixed rules, escalation never changed a verdict, and cards and profile-guided retrieval
did not either (they changed only the wording of the missing-condition list on two questions: a capital letter, and one extra wrong "task" entry for B-AI-28 that appears when cards are on;
the number of questions with the right condition named stays 17). The paper must not claim an accuracy gain from the agents on this data; any other benefit (flexible planning, robustness to wording) is untested.

### 4.5 Claim 3: does Stage 7 tell an explained difference from a genuine contradiction? (key C, 50 pairs)

**Verdict: Stage 7 raises far fewer false conflicts than plain NLI and somewhat fewer than the Gemma baseline, but it is not more accurate than the Gemma baseline, and on accuracy alone it loses to "always say explained".**
The key has 41 explained, 0 genuine and 9 not comparable pairs, so always answering "explained" would score 82 % accuracy (macro-F1 30 %).

| System | accuracy % (95 % CI) | macro-F1 % | EXPLAINED precision / recall % | NOT COMPARABLE precision / recall % | pairs called GENUINE (the key has none) | condition named right % |
|---|---|---|---|---|---|---|
| Stage 7 | 62.0 (48-74) | 56.0 | 93.1 / 65.8 | 28.6 / 44.4 | 7 of 50 | 26 |
| Gemma 12B with the conflict taxonomy | 60.0 (46-72) | 40.5 | 90.9 / 73.2 | 0.0 / 0.0 | 13 of 50 | 13 |
| Plain NLI (a contradiction is a conflict) | 0.0 (0-7) | 0.0 | - / 0.0 | 0.0 / 0.0 | 48 of 50 | - |
| Always "explained" (reference) | 82.0 | 30.0 | | | 0 of 50 | - |

Where Stage 7 errs: of the 41 explained pairs it calls 27 explained, 4 genuine and 10 not comparable; of the 9 not-comparable pairs it calls 4 right, 2 explained and 3 genuine.
The condition it names is exactly right for 26 % of the pairs both call explained (mean overlap 51 %); the Gemma baseline gets 13 % (33 %).
Plain NLI says "conflict" for 48 of 50 pairs, which is the problem Stage 7 was built to remove: the false-conflict rate falls from 96 % to 14 %. The Gemma baseline is at 26 %.
Stage 7's pairing code was not changed by the fixes of section 6 (the 50 pairs get the same 50 verdicts).

What the pairs showed about the papers (useful for the paper's discussion): DistilBERT's MRPC and QQP numbers are means of accuracy and F1, so comparing them with accuracy-only numbers is a metric mismatch;
the "Human" rows of the SQuAD 2.0 paper are an error analysis, not human performance; several pairs compare a subset with a total (RACE middle vs high school, MMLU STEM, one MLQA language vs the average).
**The limit that matters most:** the sample contains no genuine contradiction, so Stage 7's ability to *find* one is untested. The test shows false alarms and condition naming only.

### 4.6 Claim 4: do the profile parts hurt answer quality? (key B, 9 answerable questions with numbers)

**Verdict: no difference (parity), and both pipelines are far above the language model alone; the sample is too small for more.**
An answer counts as correct when it states the numbers of the key's facts (number matching, not RAGAS; see section 7). "Claims supported" is the share of the answer's claims that the NLI critic of Stage 9 accepts.

| System | correct / 9 (95 % CI) | numbers named per answer | claims supported % | scope warnings | seconds / question |
|---|---|---|---|---|---|
| Full pipeline | 5 = 55.6 % (27-81) | 6.8 | 85.2 | 1 | 10.4 |
| Without the profile stages (C / 6 / 7) | 5 = 55.6 % (27-81) | 4.0 | 91.7 | - | 7.4 |
| Language model alone, no retrieval | 0 = 0.0 % (0-30) | 0.7 | - | - | 0.9 |

Question by question, full vs no profile stages: 4 right in both, 1 only with the profile stages, 1 only without, 3 wrong in both (sign test p = 1.0). Full vs the model alone: 5 against 0, p = 0.0625 (with 9 questions even 5-0 does not reach 0.05).
The profile stages add about 3 seconds and more numbers per answer, and the critic supports a slightly smaller share of the claims (85 % vs 92 %). The frozen run gave the same 5 / 5 / 0.

## 5. Repeatability

The language-model parts are nearly, but not perfectly, repeatable, so a result should be shown with how often it was repeated.
- **Main test** (Stages 1-6 on the 30 questions), three runs of the same code: the scope decisions were identical every time (19 of 19 warnings, 1 false alarm). Stage 1's conditions differed in one run, on one question
  (B-AI-28: the language model returned the language "Tamil" instead of "Hindi" and "Tamil"), which moved 90 of 101 correct conditions to 89 of 101.
- **Baseline comparison**, two runs: no decision of any of the six systems differed on any of the 30 questions.
- **Latency**, undisturbed runs: median 3.8 to 4.4 s per question for Stages 1-6 (maximum about 10 s) on this machine.
- **The double start (section 3, item 7):** two runs shared the GPU for six minutes. The decisions were not affected, but the median time of that main test rose to 9.5 s and the timings of the comparison were inflated.
  The tables use the undisturbed repeats; the files of the disturbed runs are kept privately.

## 6. Fixes made after the error analysis (not held-out)

After the first gold run the failures were read and three general bugs were fixed, each with a unit test (197 tests pass):
1. a parameter count after a model name is its size ("LLaMA 65B" is LLaMA plus 65B; "1.3B" is not "13B");
2. task words in "-ing" form match the noun ("translating" = "translation"; also summarizing, classifying);
3. a metric word ("accuracy", "F1", ...) that the language model put in the task field is not a task.

Effect on the 30 gold questions: Stage 6 false alarms 3 to 1 (precision 86.4 to 95.0 %, F1 92.7 to 97.4 %, recall stays 100 %); Stage 1 condition precision 86.5 to 95.7 %. These fixes were made after seeing these very questions fail,
so the after-fix numbers are **not held-out**. The frozen numbers (commit `ca007a6`) are the headline. The one false alarm left (IndicCOPA, Tamil) is an extraction failure, not a matching bug
(Table 17 of 2212.05409 has two stacked blocks and the second block was stored with the first block's header). The "30 September" version of Stage 6 in the baselines table also uses the fixed matcher, which is why its row differs between the frozen and the clean table.

## 7. What these numbers can and cannot show

- **Independence.** One AI labeller who also helped build the system. The labels are checked against the PDFs, but a second reader could disagree on borderline cases (Stage 7's "not comparable" class and the lenient / strict question in job A are the soft spots).
- **Small samples.** 30 questions, 9 answerable questions with numbers, 50 pairs, 170 profiles. Confidence intervals (Wilson, 95 %) are given and are wide: one question moves the Stage 6 F1 by about three points. A difference of one or two questions is not a difference.
- **Stage 7 has no genuine contradiction in its sample.** The 28 papers are on different tasks, and the pairs came from the numeric trigger; the key has 0 genuine pairs. So the test measures false alarms and the condition named, **not** the ability to find a real contradiction. Accuracy alone is a weak yardstick: always answering EXPLAINED would score 82 %.
- **"No effect" in the ablation means "not detected on 30 questions".** The agents, escalation, cards and profile-guided retrieval were built for harder or larger question sets and for latency; this set cannot see them.
- **The answer-quality test is small and strict.** The key lists the numbers of one paper; an answer that cites another paper's number counts as a miss.
- **Not done:** the MetaLead-style extraction baseline; RAGAS (not installed; the plan is the same questions with a Gemma judge, labelled "not independent"); a human second labeller; the user interface has never been looked at in a browser.
- **Earlier silver tables** (3 October: Stage 6 100 %, answers 40 / 40) were built from the same store the system reads and are partly circular. Never quote them next to these numbers.
- **Known extraction failure:** a result table that is made of two stacked blocks can be stored with the first block's header (IndicCOPA, Table 17 of 2212.05409: wrong languages, Tamil missing). It causes the one false scope warning that is left after the fixes.

## 8. How to reproduce, and how to swap in human labels

```
# answer keys (private files under data/labelling/, only eval/labels/ is public)
python scripts/score_labels.py A data/labelling/done/A_profile_check_Shashank_AIDRAFT-claude_DONE.xlsx --out eval/labels/job_A_ai_lenient.json
python scripts/score_labels.py B data/labelling/done/B_questions_AIannotated_DONE.xlsx        # writes eval/labels/questions_gold.jsonl
python scripts/score_ai_gold_c.py data/labelling/done/C_result_pairs_AIannotated_DONE.xlsx      # writes eval/labels/job_C_ai_scores.json
# system runs (Ollama up, no other pipeline process, ~45 minutes in all)
python scripts/eval_questions.py eval/labels/questions_gold.jsonl --tag final_main              # and --set KEY=VALUE for each ablation
python scripts/eval_scope_baselines.py --gold eval/labels/questions_gold.jsonl --out eval/labels/scope_baselines_gold.final.json
python scripts/eval_answers.py --gold-b eval/labels/questions_gold.jsonl --out eval/labels/answer_quality_gold_b.final.json
python scripts/ablate_stage6.py --configs "+escalation"                                          # silver regression, must stay 32 / 32
```

If teammates (or anyone else) ever deliver real label files, put the workbooks into `data/labelling/done/` and run the same scorers
(`score_labels.py status|A|B|C`); for job C two files give Cohen's kappa. The frozen files are `eval/labels/*.frozen_ca007a6.json` and the `v2_main` / `no_*` result files;
the clean re-run is `*.final*`. Raw results, keys and freeze records are in `eval/labels/`.
