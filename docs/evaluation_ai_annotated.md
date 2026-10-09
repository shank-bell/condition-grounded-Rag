# Evaluation against AI-annotated answer keys (draft of 9 October 2026)

**Read this first.** No human ever labelled anything for this project: the teammates' label files never arrived. With the project lead's
approval (7 October) the answer keys were written by an AI assistant (Claude) that read the original PDFs. That means:

- **one main labeller** (Claude). The only second opinions are another AI (Gemini 3.1 Pro, web app, the model name as given by the project lead) on all 100 pairs of job C (two sets of 50) and a second pass of the same assistant on 20 shared rows of job A, so the only agreement scores that exist are between AIs;
- the labeller **also helped to build the system** it is judging, so it is not independent of the builder;
- the project design asked for human labels (for Stage 7: two humans, kappa of at least 0.6). That did not happen, and the paper has to say so.

Every number in this document is therefore "**against an AI-annotated key**". The human route stays open: real label files go through
the same scoring scripts without any change (section 8).

All numbers come from one PC (RTX PRO 4000, 24 GB; answer model Gemma 12B, the two agents Gemma E2B). The architecture document says
RTX 4050 6 GB, so the paper needs a footnote on the machine.

## The results in one page

"Frozen" = code of commit `ca007a6`, run before any failure was looked at (the headline). "After fixes" = the same code plus three general bug fixes made after reading the failures (not held-out; section 6). The Stage 7 rows also use a **second, held-out set of 50 pairs** (9 October): labelled blind to the system and scored once, after the Stage 7 code was frozen (section 4.5c).

| Claim | What the data say | How sure |
|---|---|---|
| 1. Extraction | 97 % of 320 sampled profiles are real results and the numbers in them are right 99.7 % of the time. Names are weaker (F1: task 78, model 85, dataset 87). Field F1 89.9 % lenient / 85.2 % strict. About half of the profiles are completely right (50 % lenient, 37 % strict); hard papers are worse (43 % / 26 %). A second sheet of 170 rows gave the same picture as the first (F1 89.4 vs 90.5), and the 20 profiles that are on both sheets got the same verdict 19 times out of 20 (kappa 0.91; one labeller twice). | 320 profiles, CI about +-5.5 points. No baseline. |
| 2. Stage 6 scope warning | Warns on 19 of 19 questions that need it, falsely on 1 of 11 after the fixes (3 of 11 frozen); F1 97.4 after fixes, 92.7 frozen. A simple LLM judge scores F1 91.4: a tie on this sample (question by question p = 0.63), but the judge answers only yes / no and misses 3 of 19, while Stage 6 names the missing condition (17 of 19 right). Plain RAG and abstain-on-weak-retrieval never warn. | 30 questions: one question moves F1 by about 3 points. |
| 2b. Which parts matter | Only the joint-coverage check: without it F1 falls to 48. The two LLM agents (as fixed rules), escalation, retrieval cards and profile-guided retrieval change no decision on 30 questions. | "Not detected at this size", not "useless". |
| 3. Stage 7 | **Held-out test (50 new pairs, scored once, section 4.5c): 58 % accuracy (CI 44-71), the same as on the first 50; the Gemma 12B judge 66 % (no significant difference, p = 0.50); "always explained" 82 %. Over all 100 pairs Stage 7 wrongly calls a pair a real contradiction 7 times of 98 (7 %), the Gemma judge 21 of 98 (21 %, p = 0.003), plain NLI 95 of 98. It found 0 of the 2 real contradictions (n = 2). Three rules added after the first 50 pairs lifted accuracy there from 58 to 70 %, but changed none of the 50 held-out verdicts (section 6).** First 50 pairs: calls 7 of 50 pairs GENUINE where the key has none, against 13 for the Gemma baseline and 48 for plain NLI. But accuracy is 62 % (Gemma baseline 60 %, "always explained" 82 %), so it is not more accurate than the baseline. A second AI labeller (Gemini 3.1 Pro, all 50 pairs) agrees with the first on 80 % (kappa 0.40, below the 0.6 target); on the settled key Stage 7 scores 58.0 % and the Gemma baseline 60.0 %. | 100 pairs in two sets, **only 2 genuine pairs** (both low confidence for one labeller): finding a real contradiction is essentially untested. |
| 4. Answer quality | 8 of 9 correct with and without the profile stages, 0 of 9 for the language model alone (p = 0.008), after correcting three key entries that asked for one table cell but listed a whole row (before the correction: 5 / 5 / 0). The correction was made after seeing the results: **not held-out**. The profile stages neither help nor hurt answer correctness (parity). | 9 questions; CI for 8 of 9 is 57-98 %. |

**Honest reading for the paper.** The condition store works as a scope checker, and the one idea that carries the result is the joint-coverage check. The data do not show that the LLM agents or Stage 7 beat simple alternatives **on accuracy**, and claims should be worded to that. What Stage 7 does show, on 100 pairs and with a paired test, is that it raises about a third as many false "real contradiction" alarms as an LLM judge (7 % against 21 %) and names the condition that explains a difference; its weak point is that it gives up when the stored profiles lack a model size or a setting (section 4.5c).
Everything is measured against keys made by one AI labeller; swapping in human labels is a re-run of the same scripts (section 8).

## 1. What was tested

The architecture names four claims. Each is scored against its own key.

| # | Claim | Measured as | Key (size) | Compared with |
|---|---|---|---|---|
| 1 | The Condition Profile Extractor stores correct profiles | per-field precision / recall / F1 and the share of fully correct profiles | A: 320 random stored profiles from 10 papers (two sheets of 170 rows, 20 shared) | nothing (the planned MetaLead-style baseline was not built) |
| 2 | Stage 6 warns when, and only when, a named condition is not covered | precision / recall / F1 of "warning given", right condition named | B: 30 questions, 19 should warn, 11 should not | plain RAG, abstain-on-weak-retrieval, an LLM judge (Sufficient-Context style, Gemma 12B), the 30 September version of Stage 6, and six ablations |
| 3 | Stage 7 tells an explained difference from a genuine contradiction | accuracy, macro-F1, false-conflict count, condition named | C: 50 result pairs, plus 50 held-out pairs | plain NLI, Gemma 12B with the conflict taxonomy |
| 4 | The profile parts do not hurt answer correctness | an answer is correct when it states the key's numbers | B: the 9 answerable questions that have numeric facts | pipeline without the profile stages (C / 6 / 7), LLM alone |

Stage 1 (question understanding) is scored on the same 30 questions: intent, complexity and the conditions it extracts.

## 2. How the keys were made

**Job A, profile check (two sheets of 170 rows, 320 different profiles).** A random sample of stored profiles from 10 papers: 6 "easy" papers (BERT, SQuAD 2.0, XNLI, XLM-R, GLUE, MuRIL) and 4 "hard" ones (T5, GPT-3, LLaMA, Llama 2).
The first sheet (4 Oct) has 99 easy and 71 hard profiles, the second (8 Oct) 105 easy and 65 hard; the last 20 rows of both sheets are the same 20 profiles, so there are 320 different ones (192 easy, 128 hard). For each profile the labeller looked at the
table row on the PDF page (49 pages per sheet, each read as a picture) and gave a verdict: ALL_OK, SOME_WRONG (with the wrong or missing fields), or NOT_A_RESULT. A cell that is true but too vague (for example setting "fine-tuned" where
the block says "translate-train-all", or a group / subject column that is not recorded) is marked `imprecise`; it is counted as right in the **lenient** reading and as wrong in the **strict** reading, and both are reported.
A wrong cell counts as one false positive and one false negative; a missing cell as one false negative. Rows 21-170 of the first sheet were marked without any other AI's answer for them (none ever existed);
rows 1-20 were marked after seeing a Gemini draft of those 20 rows (Gemini made one clear error and was inconsistent, so that route was dropped after 20 rows). The second sheet was marked with the same reading rules (copied from the comment lines of the first pass;
its labels were not opened), with no other labeller's answers visible; the 20 shared rows are not fully blind (section 3, item 8).

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

**The other files (8 Oct).** To have every label file in its designed form, the four question sheets and Aditya's pair sheet were filled by the same AI assistant. These add no new judgements: the four question sheets hold the 30 questions of job B
(8 / 8 / 7 / 7, matched to the planned type of each slot; the ids stay B-AI-xx because the template ids B-AD-xx are the same for Adarsh and Aditya), and the standard scorer run on the four files gives exactly the published key; Aditya's pair sheet holds the
first-pass labels of job C in Aditya's row order and parses to the same 50 labels. Tarun's pair sheet was filled by a different AI model, **Gemini 3.1 Pro** (web app, 8 Oct 2026). It got only the text of the blank sheet and a written task (instructions and 5 pairs per message) and never saw the first-pass labels or the system's verdicts.
It labelled all 50 pairs (the last batch had to wait until the tool's usage limit reset); whether the PDFs were attached to the chat is not recorded. Its answers and the third look are in section 4.5b.

**Job C, held-out pairs (9 Oct).** After Stage 7 had been changed (section 6) a second set of 50 pairs was built so that the change could be tested on pairs it was not made from (`scripts/make_heldout_pairs.py`, seed 1016).
The mining and the sampling are those of the first set; the pool is the 232 conflicting pairs (of 365 mined) that share no pair of papers + subject and no stored profile with the first set, and the strata that steer the sample come from the *frozen* Stage 7 (every fix-A switch off), so the fixes did not choose their own test.
The 50 pairs come from 25 pairs of papers. The system's verdicts (frozen and current) went into a private file that was never printed or read before the labels were final. The first pass (Claude) was made blind to the system, from the sheet's excerpts and five PDF pages read as pictures.
The second pass was made by **Gemini 3.1 Pro** (web app; the model name as given by the project lead; whether the PDFs were attached is not recorded) from the blank sheet text, in 10 batches of 5 pairs with the same written instructions as for the first set. Details and results: section 4.5c.

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
8. **Second job-A sheet (8 Oct): the 20 shared rows are not fully blind.** The earlier labels were not opened, but before the pass two lists from them had been printed in the session (the row numbers it tagged `imprecise`, and its three NOT_A_RESULT rows 58, 151, 149);
   row 151 is a shared row and got NOT_A_RESULT on both sheets. The one disagreement among the 20 shared rows (a GPT-3 translation score whose direction, En to Ro, is not recorded) is the lenient / strict policy question itself.
9. **Same labeller, same rules.** The agreement between the two job-A sheets measures how stable one labeller is, not whether two people agree.
10. **Second labeller, first attempt (8 Oct).** The first paste into Gemini had no instructions and the model only copied the table fields; that answer was discarded. The files were rebuilt as one text with the instructions, an explicit task line and a reminder
    at the end; the later answers have the right format (checked line by line by a script).
11. **The third look is not blind (8 Oct).** The 10 pairs where the first pass and Gemini disagreed were settled by the same assistant that made the first pass, after seeing both labels, with the PDF pages and the stored profiles' fields (not the system's verdicts).
    It sided with its own first pass on 8 of the 10, which is what a non-blind tie-break tends to do; the settled key is therefore used only as a sensitivity analysis and the frozen first-pass key stays the headline.
12. **What the second labeller found in the first pass:** two wrong verdicts (P19 and P29). In both the first pass had seen that the number in the sheet was not what the table says (a margin read as a score; a SQuAD score in an MNLI slot) but still judged the pair on those numbers.
13. **Stage 7 fix A was tuned on the first 50 pairs (9 Oct).** Three rules were added after reading the 21 wrong pairs of the first set; accuracy there rose from 58 to 70 %. That gain is not held-out and it did not carry over: on the held-out 50 pairs the rules changed no verdict (section 6).
    Two further ideas were tried on the first set and dropped: filling blank model sizes in the stored profiles (it would have turned "cannot tell" into false genuine calls) and a fourth rule that broke 7 pairs that were right.
14. **Held-out pairs, third look not blind (9 Oct).** The two labellers disagreed on 7 verdicts and on the table-reading flag of 5 more pairs. The same assistant that made the first pass settled these 12 after seeing both labels (not blind to the labels, still blind to the system):
    it sided with its own first pass on 6 of the 7 verdicts and with Gemini on 1 (H05, a "+1.4 %" gain that is not a score; the same case as H36, where both labellers already said not comparable). The first-pass labelling was inconsistent there and the settled key follows Gemini.
    The label is the soft spot, so the key is also scored without its 6 low-confidence pairs (section 4.5c).
15. **The labelling sheet itself misled the second labeller on some held-out pairs.** The excerpt of a table row does not show the block heading above it: for XLNet's Table 5 it dropped "Multi-task ensembles on test", so a test-ensemble number looked like a single-model dev number (H26: Gemini said genuine, the page says explained; H41: only the table-reading flag was affected).
    And for one stored profile the excerpt showed another cell with the same value (88.5) than the profile came from (H44: Gemini said not comparable and flagged a table-reading error; H07: only the flag). The third look checked these on the PDF page and in the stored profile; they are limits of the sheet, not of the system.
    Gemini flagged 9 table-reading errors against the first pass's 6; the third look kept 6.
16. **Answer-quality key corrected after the first scoring (9 Oct).** Three of the four wrong answers were scorer artefacts: the key's facts text listed a whole table row (all shot settings, EM and F1) while the question asked for one cell (5-shot, zero-shot, F1), and the scorer wants half of all listed numbers.
    The three expected values were narrowed to the asked cell for all systems alike (section 6); this is a change to the key made after seeing the results, so the corrected numbers are not held-out and the original ones are kept next to them.
17. **Convention for the third looks.** When the two numbers are not the same quantity or the same system (a margin, a human row, a wrong cell, a mean of two metrics), NOT COMPARABLE takes priority over EXPLAINED, even if the conditions also differ. The first set was labelled the same way, but the rule was written down only now.

## 4. Results

All numbers are against the AI-annotated keys. "95 % CI" is a Wilson interval and shows how little 30 questions or 50 pairs can say.
"Frozen" means the code of commit `ca007a6`, run before any failure was looked at. "Clean re-run" means the code of commit `4d06f84` (the same code plus the three fixes of section 6), run again
from one commit on 8 October; those numbers are "after error analysis, not held-out".

### 4.1 Claim 1: are the stored profiles right? (key A, 320 profiles on two sheets)

**Verdict: the numbers are right, the names are the weak part, and about half of the profiles are completely right.**
311 of the 320 sampled profiles are real results (97.2 %). Of those, 157 are completely right (50.5 %, 95 % CI 45-56) when a true-but-vague cell counts as right (lenient),
and 115 (37.0 %, CI 32-42) when it counts as wrong (strict). On the 6 easy papers (186 real results) 55.4 % / 44.6 % are completely right, on the 4 hard papers (T5, GPT-3, LLaMA, Llama 2; 125 real results) 43.2 % / 25.6 %.
The two sheets agree: the first (170 rows) gave 52.7 % / 40.7 % completely right and field F1 90.5 % / 85.8 %, the second (170 rows, 150 of them new) 48.5 % / 33.1 % and F1 89.4 % lenient.

| Field | precision % | recall % | F1 % lenient | F1 % strict | F1 % easy papers (lenient) | F1 % hard papers (lenient) |
|---|---|---|---|---|---|---|
| value | 99.7 | 99.7 | 99.7 | 99.7 | 100.0 | 99.2 |
| language | 97.7 | 97.7 | 97.7 | 95.7 | 97.4 | 98.1 |
| metric | 96.1 | 96.1 | 96.1 | 95.8 | 98.4 | 92.8 |
| setting | 88.2 | 88.2 | 88.2 | 72.2 | 86.7 | 90.4 |
| model_size | 91.9 | 91.0 | 91.5 | 90.5 | 90.3 | 92.0 |
| dataset | 87.9 | 86.2 | 87.0 | 77.6 | 86.9 | 87.2 |
| model | 85.2 | 85.2 | 85.2 | 79.1 | 94.6 | 71.2 |
| task | 78.0 | 78.0 | 78.0 | 77.3 | 80.7 | 74.0 |
| dataset_version | 80.8 | 61.8 | 70.0 | 70.0 | 72.7 | 40.0 |
| **all fields (micro)** | 90.2 | 89.6 | **89.9** | **85.2** | 91.6 | 87.6 |

How to read it: the 9 fields are scored on every real result; a wrong cell is one false positive and one false negative. Almost every error is in a *name*, not in a number:
a data set name in the task field (36 times on the second sheet alone); a cut or garbled model name ("lama 1", "lama 2hat", "L C", "aLLMA", "Fln", "Wikipedia"); "Dev Set X" or a language pair stored as the data set; a pre-training data set or objective stored as the model (T5);
a demographic group stored as the language (Llama 2, ToxiGen); two stacked table blocks mixed up (GPT-3 SuperGLUE, IndicCOPA); a value that belongs to another model of the same row (MMLU Table 16 of LLaMA); the right number under the wrong metric (NER scored with F1 stored as accuracy);
scores on an augmented data set stored as plain SQuAD 1.1 test (Table 4 of the SQuAD 2.0 paper); a group or subject column not recorded (MMLU, ToxiGen, BOLD, WinoGender; tagged `imprecise`). Hard papers are worse because of the names: model F1 is 94.6 on the easy and 71.2 on the hard papers.
`dataset_version` is found in 21 of the 34 cases that have one (recall 61.8 %) and it is stored for only 4.4 % of all profiles. The setting field is where lenient and strict differ most (88.2 vs 72.2), because many settings are true but vague.
**Repeat check:** the 20 profiles that are on both sheets got the same verdict 19 times out of 20 (Cohen's kappa 0.91; 97.2 % of the 180 cell marks equal, 95.6 % in the strict reading); the one difference is the translation direction of a GPT-3 score, which is the lenient / strict question.
This is one labeller twice (section 3, item 9), so it shows that the labelling is stable, not that two people agree. There is **no baseline** for this claim: the MetaLead-style extractor named in the plan was not built.

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

### 4.5 Claim 3: does Stage 7 tell an explained difference from a genuine contradiction? (key C, first 50 pairs; the held-out 50 are in 4.5c)

**Verdict: Stage 7 raises far fewer false conflicts than plain NLI and somewhat fewer than the Gemma baseline, but it is not more accurate than the Gemma baseline, and on accuracy alone it loses to "always say explained".**
The key has 41 explained, 0 genuine and 9 not comparable pairs, so always answering "explained" would score 82 % accuracy (macro-F1 45 %, averaged over the two classes that occur, like the other rows).

| System | accuracy % (95 % CI) | macro-F1 % | EXPLAINED precision / recall % | NOT COMPARABLE precision / recall % | pairs called GENUINE (the key has none) | condition named right % |
|---|---|---|---|---|---|---|
| Stage 7 | 62.0 (48-74) | 56.0 | 93.1 / 65.8 | 28.6 / 44.4 | 7 of 50 | 26 |
| Gemma 12B with the conflict taxonomy | 60.0 (46-72) | 40.5 | 90.9 / 73.2 | 0.0 / 0.0 | 13 of 50 | 13 |
| Plain NLI (a contradiction is a conflict) | 0.0 (0-7) | 0.0 | - / 0.0 | 0.0 / 0.0 | 48 of 50 | - |
| Always "explained" (reference) | 82.0 | 45.1 | | | 0 of 50 | - |

Where Stage 7 errs: of the 41 explained pairs it calls 27 explained, 4 genuine and 10 not comparable; of the 9 not-comparable pairs it calls 4 right, 2 explained and 3 genuine.
The condition it names is exactly right for 26 % of the pairs both call explained (mean overlap 51 %); the Gemma baseline gets 13 % (33 %).
Plain NLI says "conflict" for 48 of 50 pairs, which is the problem Stage 7 was built to remove: the false-conflict rate falls from 96 % to 14 %. The Gemma baseline is at 26 %.
Stage 7's pairing code was not changed by the fixes of section 6 (the 50 pairs get the same 50 verdicts).

What the pairs showed about the papers (useful for the paper's discussion): DistilBERT's MRPC and QQP numbers are means of accuracy and F1, so comparing them with accuracy-only numbers is a metric mismatch;
the "Human" rows of the SQuAD 2.0 paper are an error analysis, not human performance; several pairs compare a subset with a total (RACE middle vs high school, MMLU STEM, one MLQA language vs the average).
**The limit that matters most:** the sample contains no genuine contradiction, so Stage 7's ability to *find* one is untested. The test shows false alarms and condition naming only.

### 4.5b Job C with a second AI labeller and a settled key (sensitivity analysis, all 50 pairs)

**Verdict: the two AI labellers agree on 80 % of the pairs, the agreement score (kappa 0.40) is below the 0.6 target, and a settled key tells the same story about Stage 7 as the frozen key.**
Gemini 3.1 Pro labelled all 50 pairs. First pass against Gemini: 40 of 50 verdicts are the same (80 %), Cohen's kappa 0.40 (target 0.6 **not** met; kappa is low in part because 82 % of the pairs are "explained", which makes chance agreement high).

| first pass (rows) / Gemini (columns) | EXPLAINED | NOT COMPARABLE | GENUINE |
|---|---|---|---|
| EXPLAINED (41) | 36 | 4 | 1 |
| NOT COMPARABLE (9) | 3 | 4 | 2 |

Of the 36 pairs that both call explained, only 20 name exactly the same differing conditions: "dev vs test" appears under two names in the instructions ("dataset version / split" and "setting") and the two labellers chose differently.
Table-reading errors: the first pass flagged 7 pairs, Gemini 15 (5 in common); most of Gemini's extra flags are pairs that compare different languages, which is not a table-reading error.

**Third look.** The 10 pairs with different verdicts (P12, P13, P16, P19, P23, P26, P29, P35, P39, P40) were settled with the PDF pages by the same assistant that made the first pass (not blind, section 3 item 11): it sided with the first pass on 8 and with Gemini on 2 (P19, P29).
Gemini's three "genuine" verdicts (P13, P35, P39) all turned out differently: the SQuAD human-performance numbers (86.8 vs 91.2 F1) have no stated protocol on one side, and P35 compares a test-leaderboard ensemble with a single dev result.
The settled key has 41 explained, 9 not comparable and **0 genuine** pairs; it differs from the first-pass key on two pairs only (P19 and P29).

| Stage 7 against the settled key (50 pairs) | accuracy % | macro-F1 % | pairs called GENUINE (key: none) | condition named right % |
|---|---|---|---|---|
| Stage 7 | 58.0 | 50.2 | 7 of 50 | 35 |
| Gemma 12B with the conflict taxonomy | 60.0 | 40.5 | 13 of 50 | 27 |
| Plain NLI | 0.0 | 0.0 | 48 of 50 | - |
| Always "explained" (reference) | 82.0 | 45.1 | 0 of 50 | - |

Reading: the same story as on the frozen key (62.0 % against 60.0 %). On the settled key the Gemma baseline is two points ahead on accuracy (60.0 against 58.0, one pair apart), Stage 7 is ahead on macro-F1 (50.2 against 40.5) and raises fewer false conflicts (7 against 13 and 48), and it is the only one that gets any of the not-comparable pairs right.
On the same 50 pairs the frozen first-pass key gives 62.0 % accuracy (macro-F1 56.0) and the settled key 58.0 % (50.2): only two verdicts differ (P19 and P29), and flipping two labels moves the accuracy by 4 points. That is the size of the label noise on a set this small.

### 4.5c Claim 3 on 50 held-out pairs (the clean test, scored once)

**Verdict: on pairs that no rule was made from, Stage 7 is right on 58 % (the same as on the first 50), it is not more accurate than the Gemma judge, and the three rules of fix A changed none of the 50 verdicts. Its strength is that it rarely cries "real contradiction" when there is none; its weakness is that it gives up when a stored profile lacks a model size or a setting.**
How the set and its key were made is in section 2. Everything below is against the settled key; `scripts/score_heldout_c.py` scores it in one run (`eval/labels/job_C_heldout_scores.json`).

**The labels.** First pass against Gemini 3.1 Pro: 43 of 50 verdicts are the same (86 %), Cohen's kappa 0.52 (the 0.6 target is **not** met again; 41 of 50 pairs are "explained", which makes chance agreement high).

| first pass (rows) / Gemini (columns) | EXPLAINED | NOT COMPARABLE | GENUINE |
|---|---|---|---|
| EXPLAINED (42) | 38 | 3 | 1 |
| NOT COMPARABLE (6) | 3 | 3 | 0 |
| GENUINE (2) | 0 | 0 | 2 |

Of the 38 pairs both call explained, 29 name exactly the same conditions. A third look (not blind to the labels, blind to the system; section 3, items 14-15) settled the 7 verdict disagreements (6 for the first pass, 1 for Gemini) and the table-reading flag of 5 more pairs.
The settled key has **41 explained, 7 not comparable and 2 genuine pairs**, 6 pairs with a table-reading error and 6 labels with low confidence; it differs from the first-pass key on one pair.
The two genuine pairs are the first possible real contradictions in the corpus: OpenAI GPT's GLUE test scores in the BERT paper (RTE 56.0, CoLA 45.4) and in the MobileBERT paper (RTE 69.1, CoLA 47.2), and neither paper says why they differ (the first pass marked them low confidence, Gemini high).

**Stage 7 and the baselines against the settled key** (Gemma 12B with the same prompt as in 4.5; "frozen" = every fix-A switch off).

| System | right of 50 | accuracy % (95 % CI) | macro-F1 % (3 classes) | pairs called GENUINE (key: 2) | NOT COMPARABLE found (key: 7) |
|---|---|---|---|---|---|
| Stage 7, frozen | 29 | 58.0 (44-71) | 27.4 | 0 | 1 |
| Stage 7 with fix A | 29 | 58.0 (44-71) | 27.4 | 0 | 1 |
| Gemma 12B with the conflict taxonomy | 33 | 66.0 (52-78) | 38.6 | 9 (1 right) | 1 |
| Plain NLI | 2 | 4.0 (1-14) | 2.6 | 49 | 0 |
| Always "explained" (reference) | 41 | 82.0 (69-90) | 30.0 | 0 | 0 |

(Macro-F1 is averaged over all three classes because all three occur in this key, so it is not comparable with the first set's two-class average.)
Fix A against frozen: no pair differs (p = 1.0). Stage 7 against Gemma: 8 pairs only Stage 7 gets right, 12 only Gemma (p = 0.50). Without the 6 low-confidence pairs: Stage 7 65.9 %, Gemma 68.2 % (44 pairs); without the pairs with a table-reading error: 61.4 % and 70.5 %.
When both call a pair explained, Stage 7 names exactly the right conditions for 25 % of 28 pairs (mean overlap 51 %), Gemma for 35 % of 31 (51 %).

**Where Stage 7 errs (21 of 50).**
- **13 explained pairs called "not comparable"** (it gives up): in 12 a model size is recorded for only one of the two papers, typically a plain name ("RoBERTa", "XLNet") against a sized one ("RoBERTa_base", "XLNetLarge"), and in 1 a dataset version. In several of them the stored profiles also lack the real difference (test against dev, ensemble against single model), so filling the blank would not have helped.
- **6 not-comparable pairs called "explained":** a margin or a gain read as a score, a baseline and a model that carry the same name, a mean of two metrics, a wrong cell; in two of them the same model size is written once as a label and once as a count.
- **2 genuine pairs called "explained":** the profiles say "single model" in one paper and "fine-tuned" in the other, two words about different things that were read as a difference.

**All 100 pairs (first 50 with the frozen verdicts, plus the held-out 50).** The key has 98 pairs that are not a real contradiction. How often does a system call one of them a real contradiction (GENUINE)?

| System | false "real contradiction" calls | rate % (95 % CI) | real contradictions found (key: 2) | accuracy over 100 pairs |
|---|---|---|---|---|
| Stage 7 | 7 of 98 | 7.1 (3.5-14.0) | 0 | 58 (48-67) |
| Gemma 12B with the conflict taxonomy | 21 of 98 | 21.4 (14.5-30.6) | 1 | 63 (53-72) |
| Plain NLI | 95 of 98 | 96.9 (91.4-99.0) | 2 | 2 |

In the pair-by-pair comparison with Gemma, 17 pairs are called "real contradiction" only by Gemma and 3 only by Stage 7 (sign test p = 0.0026). On accuracy the two are not different: 15 pairs only Stage 7 gets right, 20 only Gemma (p = 0.50).
Two genuine pairs cannot say whether Stage 7 can find a real contradiction; it found none of them.

**Reading for the paper.** The honest claim is about false alarms and explanations, not accuracy: Stage 7 calls a real contradiction where there is none about a third as often as an LLM judge, it says which condition explains a difference, it needs no generative LLM call, and every verdict traces back to stored values.
It is not more accurate than the LLM judge, it never found a real contradiction (there are almost none in this corpus), and on accuracy alone "always explained" beats both (82 %). Its main failure is a data problem: when the stored profile has no size or setting, or has the wrong one, the rule cannot see the difference.
A real fix would be in the extraction (record the heading of a table block such as "ensembles on test", and a paper's default model size), not in Stage 7's rules; that would need a re-extraction and a new test set and was not done.

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

**The key was corrected after this scoring (9 Oct; section 3, item 16): not held-out.** The scorer counts an answer as correct when it states at least half of all the numbers in the key's "facts" text. For three questions that text listed a whole table row while the question asked for one cell, so a right answer scored wrong:
the LLaMA 65B question asks for the 5-shot value (35.0; the facts listed the 0-, 1-, 5- and 64-shot values), the GPT-3 question for the zero-shot BLEU (27.2; zero-, one- and few-shot were listed) and the DistilBERT question for an F1 (EM and F1 of two model sizes were listed).
The expected values were narrowed to the asked cell (35.0; 27.2; an F1 of 64.1 or 69.5), for all three systems alike and from the question text only; the stored answers were re-scored without a new run (`scripts/rescore_answers_gold_b.py`, `eval/labels/answer_quality_gold_b.corrected_key.json`).

| System | correct / 9 before | correct / 9 after the correction (95 % CI) |
|---|---|---|
| Full pipeline | 5 | 8 = 88.9 % (57-98) |
| Without the profile stages (C / 6 / 7) | 5 | 8 = 88.9 % (57-98) |
| Language model alone, no retrieval | 0 | 0 = 0.0 % (0-30) |

Full against the model alone: 8 questions only the full pipeline gets right, none the other way (sign test p = 0.0078). Full against no profile stages: 7 right in both, 1 only with the profile stages (the mT5-Large / XLM-R Swahili question) and 1 only without (the broad "How well does BERT perform on SQuAD 2.0?", where the full pipeline's answer names one of the four expected numbers among several from other papers' tables), none wrong in both (p = 1.0): parity.

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

**Stage 7, fix A (9 Oct; tuned on the first 50 pairs, not held-out).** After the first 50 pairs had been scored, their 21 wrong pairs (against the settled key) were read and three rules were added to `classify`, each with its own switch under `[contradiction]` in `config.toml` and a unit test (`tests/test_stage7_fix_a.py`; 203 tests pass):
1. a "Human" row is a study of people (agreement between annotators, an error analysis, a leaderboard figure), measured differently by different papers: such a pair is NOT COMPARABLE;
2. MRPC, QQP and STS-B have two official metrics (accuracy and F1), and a paper that prints one number per task may print either or their mean (DistilBERT's 86.2 for ELMo on QQP is the mean of GLUE's 88.0 and 84.3): such a pair is never called GENUINE;
3. MNLI matched / mismatched are two evaluation sets, a split difference like dev / test.

Effect on the first 50 pairs: right on 29 before and 35 after (58 to 70 %), pairs called GENUINE 7 to 3, not-comparable pairs found 3 to 8 of 9, no pair made worse (without the 9 low-confidence labels: 27 to 31 of 41); each rule alone adds 3, 2 and 1 pairs. **That gain is tuned to those pairs. On the 50 held-out pairs the rules changed no verdict** (section 4.5c): the frozen Stage 7 never called a held-out pair genuine, and the pairs it got wrong there are of a different kind.
Two ideas were tried on the first set and dropped: filling the blank model sizes and settings of the stored profiles (the real difference is usually missing too, so a filled blank would have turned "cannot tell" into false genuine calls), and a fourth rule "only call a difference when both profiles describe the same kind of label", which broke 7 pairs that were right.

**Answer key (9 Oct; corrected after the first scoring, not held-out).** Three expected-value lists were narrowed to the asked cell (section 4.6, 3 questions: 35.0; 27.2; an F1 of 64.1 or 69.5), for all systems alike. The answer-quality numbers are given before and after.

## 7. What these numbers can and cannot show

- **Independence.** One AI labeller who also helped build the system. The labels are checked against the PDFs, but a second reader could disagree on borderline cases (Stage 7's "not comparable" class and the lenient / strict question in job A are the soft spots). The second job-A sheet and the filled teammate sheets (except the second-labeller pair sheets, which are Gemini's) were made by the same labeller with the same rules; both job-C third looks are not blind and sided with the first pass on most pairs (8 of 10 and 6 of 7).
  The agreement between the two AI labellers on job C is 80 % (kappa 0.40) and 86 % (kappa 0.52), both below the 0.6 target. Gemini's PDFs-attached status is not recorded for either round; the sheet excerpts it relied on sometimes omit a table's block heading (section 3, item 15).
- **Small samples.** 30 questions, 9 answerable questions with numbers, 50 pairs, 320 profiles. Confidence intervals (Wilson, 95 %) are given and are wide: one question moves the Stage 6 F1 by about three points. A difference of one or two questions is not a difference.
- **Stage 7 has almost no genuine contradiction in its samples.** The 28 papers are on different tasks, and the pairs came from the numeric trigger; the first key has 0 genuine pairs and the held-out key 2 (OpenAI GPT's GLUE test scores, one labeller unsure). So the tests measure false alarms and the condition named, **not** the ability to find a real contradiction (0 of 2 found). Accuracy alone is a weak yardstick: always answering EXPLAINED would score 82 % on both sets.
- **The held-out result is one sample of 50 pairs.** 58 % has a 95 % CI of 44-71 %, and the Gemma judge's 66 % lies inside it; the false-alarm advantage (7 % against 21 %) is the only difference that is statistically clear (p = 0.003, 100 pairs). The held-out key itself was settled by a third look that is not blind to the labels.
- **"No effect" in the ablation means "not detected on 30 questions".** The agents, escalation, cards and profile-guided retrieval were built for harder or larger question sets and for latency; this set cannot see them.
- **The answer-quality test is small and strict.** The key lists the numbers of one paper; an answer that cites another paper's number counts as a miss. Nine questions; the corrected key (8 of 9) was fixed after the first scoring (section 3, item 16), so it is not held-out.
- **Not done:** the MetaLead-style extraction baseline; RAGAS (not installed; the plan is the same questions with a Gemma judge, labelled "not independent"); a human second labeller; the user interface has never been looked at in a browser.
- **Earlier silver tables** (3 October: Stage 6 100 %, answers 40 / 40) were built from the same store the system reads and are partly circular. Never quote them next to these numbers.
- **Known extraction failure:** a result table that is made of two stacked blocks can be stored with the first block's header (IndicCOPA, Table 17 of 2212.05409: wrong languages, Tamil missing). It causes the one false scope warning that is left after the fixes.

## 8. How to reproduce, and how to swap in human labels

```
# answer keys (private files under data/labelling/, only eval/labels/ is public)
python scripts/score_labels.py A data/labelling/done/A_profile_check_Shashank_AIDRAFT-claude_DONE.xlsx --out eval/labels/job_A_ai_lenient.json
python scripts/score_labels.py B data/labelling/done/B_questions_AIannotated_DONE.xlsx        # writes eval/labels/questions_gold.jsonl
python scripts/score_ai_gold_c.py data/labelling/done/C_result_pairs_AIannotated_DONE.xlsx      # writes eval/labels/job_C_ai_scores.json
python scripts/fill_job_a_from_ai.py --sheet data/labelling/A_profile_check_Adarsh.xlsx --answers data/labelling/claude_A_answers_adarsh.txt --tag claude-adarsh --source "..."   # --strict for the strict workbook
python scripts/score_labels.py A <first sheet done file> <second sheet done file> --out eval/labels/job_A_ai_both_lenient.json    # 320 profiles + agreement on the 20 shared rows
python scripts/score_labels.py B data/labelling/done/B_questions_{Adarsh,Aditya,Shashank,Tarun}_AIannotated_DONE.xlsx --out <file>   # the four sheets reproduce questions_gold.jsonl
python scripts/fill_job_c_from_ai.py --answers "data/labelling/ai_fiesta/answers_gemini_b*.jsonl" --model "Gemini 3.1 Pro (web app)" --date "8 Oct 2026" --pdfs unknown --tag gemini31pro   # Tarun's sheet from the pasted answers
python scripts/score_labels.py C --first <Aditya pass-1 file> --second <Tarun Gemini file> --third <third-look file> --out eval/labels/job_C_ai_settled45_scores.json   # agreement, settled key, Stage 7 and the baselines against it
# held-out pairs for Stage 7 (9 Oct): build, label (first pass / Gemini / third look), baselines, ONE scoring run
python scripts/make_heldout_pairs.py --pairs 50 --seed 1016            # blank sheets data/labelling/C_heldout_pairs_{first,second}.xlsx + a private key with the system's verdicts (never print it before the labels are final)
python scripts/fill_job_c_from_ai.py --answers <answers .jsonl> --sheet data/labelling/C_heldout_pairs_first.xlsx --model "..." --date ... --tag heldout-first --role "HELD-OUT FIRST PASS"   # the same for the Gemini answers (sheet ..._second) and the third look
python scripts/run_pair_baselines.py --key data/labelling/private/job_C_heldout_key.json --out data/labelling/private/job_C_heldout_baselines.json    # needs Ollama; ~75 s
python scripts/score_heldout_c.py                                      # frozen Stage 7, fix A, plain NLI, Gemma, "always explained", and the pooled 100-pair false-alarm numbers -> eval/labels/job_C_heldout_scores.json
python scripts/rescore_answers_gold_b.py                               # answer-quality numbers with the corrected key (no run) -> eval/labels/answer_quality_gold_b.corrected_key.json
# system runs (Ollama up, no other pipeline process, ~45 minutes in all)
python scripts/eval_questions.py eval/labels/questions_gold.jsonl --tag final_main              # and --set KEY=VALUE for each ablation
python scripts/eval_scope_baselines.py --gold eval/labels/questions_gold.jsonl --out eval/labels/scope_baselines_gold.final.json
python scripts/eval_answers.py --gold-b eval/labels/questions_gold.jsonl --out eval/labels/answer_quality_gold_b.final.json
python scripts/ablate_stage6.py --configs "+escalation"                                          # silver regression, must stay 32 / 32
```

If teammates (or anyone else) ever deliver real label files, put the workbooks into `data/labelling/done/` and run the same scorers
(`score_labels.py status|A|B|C`); for job C two files give Cohen's kappa. The frozen files are `eval/labels/*.frozen_ca007a6.json` and the `v2_main` / `no_*` result files;
the clean re-run is `*.final*`. Raw results, keys and freeze records are in `eval/labels/`.
