# Evaluation against AI-annotated answer keys (draft of 9 October 2026, updated 11 October)

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
| 1. Extraction | 97 % of 320 sampled profiles are real results and the numbers in them are right 99.7 % of the time. Names are weaker (F1: task 78, model 85, dataset 87). Field F1 89.9 % lenient / 85.2 % strict. About half of the profiles are completely right (50 % lenient, 37 % strict); hard papers are worse (43 % / 26 %). A second sheet of 170 rows gave the same picture as the first (F1 89.4 vs 90.5), and the 20 profiles that are on both sheets got the same verdict 19 times out of 20 (kappa 0.91; one labeller twice). **11 Oct (sections 4.1b and 4.1c): a code-only repair of the stored profiles (raw rows kept, no model called) lifts the share of completely right profiles on 94 results of 34 papers of an external benchmark, drawn after the code was frozen, from 37 % to 61 % (strict 22 to 39 %; 23 fixed, 1 broken, p < 0.0001) and the field F1 from 90.3 to 94.0 (strict 84.9 to 90.0); the completely-right share stays below 85 %. On the public MetaLead benchmark (3,568 human-annotated results, 43 papers) the pipeline finds 88.4 % of the results against 79.0 % for the same model with the benchmark's own prompt on the raw text, and agrees with the annotators' data set, metric and task names for 54.3 % of the results it finds (plain LLM 38.3 %; repaired 62.8 %, a development number).** | 320 profiles, CI about +-5.5 points; 94 clean results for the repair (CI of the 61 %: 50-70); a baseline only on the benchmark, scored with a looser matcher of mine. |
| 2. Stage 6 scope warning | **11 Oct (section 4.7c, repaired store and three fixes; development numbers): 19 / 1 / 0, F1 97.4, condition named 19 of 19; the silver regression stays 32 of 32.** Warns on 19 of 19 questions that need it, falsely on 1 of 11 after the fixes (3 of 11 frozen); F1 97.4 after fixes, 92.7 frozen. A simple LLM judge scores F1 91.4: a tie on this sample (question by question p = 0.63), but the judge answers only yes / no and misses 3 of 19, while Stage 6 names the missing condition (17 of 19 right). Plain RAG and abstain-on-weak-retrieval never warn. | 30 questions: one question moves F1 by about 3 points. |
| 2b. Which parts matter | Only the joint-coverage check: without it F1 falls to 48. The two LLM agents (as fixed rules), escalation, retrieval cards and profile-guided retrieval change no decision on 30 questions. | "Not detected at this size", not "useless". |
| 3. Stage 7 | **11 Oct (section 4.5f; stored profiles repaired, policy B on): 220 of 250 labelled pairs right (88.0 %, CI 83-92) against 206 (82.4 %) for table grounding alone, 201 (80.4 %) for "always EXPLAINED" and 160 (64 %) for the Gemma 12B judge. On the third fresh set (50 new pairs, clean) 46 right (92 %) against 42 for grounding alone, 45 for "always EXPLAINED" and 31 for Gemma; macro-F1 51 against 46, 32 and 33. The clean evidence for the policy is small (5 gained, 1 lost, p = 0.22); the shipped Stage 7 calls a pair GENUINE falsely 5 times in 250 (Gemma 12B: 40 times) and finds none of the 3 real contradictions of the five sets.** **With table grounding (10 Oct evening, section 4.5e; 50 new pairs built and labelled blind after the design was frozen, scored once): 44 of 50 right (88 %) against 34 (68 %) without it (11 gained, 1 lost, p = 0.006); the Gemma 12B judge 34 (68 %); "always EXPLAINED" 39 (78 %, the difference p = 0.23); macro-F1 83 against 60 / 42 / 44; 8 of the 11 not-comparable pairs found (both baselines 0); no false genuine call (Gemma 6 of 50). Over all 200 labelled pairs 82.5 % right against 65.5 % before; false genuine calls 2 % against 17 % for Gemma. The cause of the earlier mistakes was the stored cards (a block heading that never reached its rows, a lost split, a row stored under another system's name), read again from the cards' own tables.** Before that, a fresh test (10 Oct, section 4.5d; 50 new pairs labelled blind by one AI labeller, scored once): 66 % accuracy (CI 52-78; "always EXPLAINED" 70 %, Gemma 64 %), macro-F1 57 against 41 for the Gemma judge and for "always EXPLAINED", no false genuine call (Gemma 6 of 50), 5 of the 15 not-comparable pairs found (0 and 0). Over all 150 pairs the frozen Stage 7 calls a pair a real contradiction wrongly 7 times of 148 (5 %), the Gemma judge 28 times (19 %). A "one-sided condition" policy found on the first two sets lifts accuracy to the base rate (81 % over 150 pairs) only by calling nearly every pair EXPLAINED, so it is off; fix A does not generalise (it changes no verdict on the held-out or the fresh set).** Held-out test (50 new pairs, scored once, section 4.5c): 58 % accuracy (CI 44-71), the same as on the first 50; the Gemma 12B judge 66 % (no significant difference, p = 0.50); "always explained" 82 %. Over all 100 pairs Stage 7 wrongly calls a pair a real contradiction 7 times of 98 (7 %), the Gemma judge 21 of 98 (21 %, p = 0.003), plain NLI 95 of 98. It found 0 of the 2 real contradictions (n = 2). Three rules added after the first 50 pairs lifted accuracy there from 58 to 70 %, but changed none of the 50 held-out verdicts (section 6).** First 50 pairs: calls 7 of 50 pairs GENUINE where the key has none, against 13 for the Gemma baseline and 48 for plain NLI. But accuracy is 62 % (Gemma baseline 60 %, "always explained" 82 %), so it is not more accurate than the baseline. A second AI labeller (Gemini 3.1 Pro, all 50 pairs) agrees with the first on 80 % (kappa 0.40, below the 0.6 target); on the settled key Stage 7 scores 58.0 % and the Gemma baseline 60.0 %. | 250 pairs in five sets, **only 3 genuine pairs** (low confidence, one labeller): finding a real contradiction is essentially untested; only the third fresh set is clean for policy B. |
| 4. Answer quality | 8 of 9 correct with and without the profile stages, 0 of 9 for the language model alone (p = 0.008), after correcting three key entries that asked for one table cell but listed a whole row (before the correction: 5 / 5 / 0). The correction was made after seeing the results: **not held-out**. The profile stages neither help nor hurt answer correctness (parity). | 9 questions; CI for 8 of 9 is 57-98 %. |
| 5. Retrieval and questions the papers cannot answer (9 Oct, section 4.7) | **11 Oct (section 4.7c, repaired store and three fixes): the gold page is in the top 5 for 78.6 % (85.7 % before the repair), among the answer's sources for 85.7 %, the right paper for 92.9 %; 100 % of the 20 out-of-corpus questions end with a warning.** Against the gold evidence pages (28 questions): the page that holds the answer is among the sources the answer is written from for **85.7 %** (covered and one-condition-missing questions 100 %, partly covered comparisons 60 %); the right paper for 96.4 %. The reranker's own top 5 held the page for 75 %; with `bge-reranker-base` (swapped in on 10 Oct, section 4.7b) it is **85.7 %** (store-made questions 87.8 to 93.1 %), but the end-to-end 85.7 % does not move. **20 of 20 questions about things that are not in the corpus end with a warning (100 %; 19 of 20 before a Stage 1 fix of 10 Oct)**; the scope warning alone catches all 20, the reranker's weak-evidence flag alone 50 % (70 % with the old reranker). | 28 and 20 questions; the 10 Oct changes were made after reading these very questions (development numbers). The earlier store-made retrieval test (language + task questions: 60 % at the top 5) was circular. |
| 6. Faithfulness and the optional rewrite loop (9-10 Oct, section 4.8) | **99 % of the decimal numbers in the shipped answers occur in the text they were written from** (120 of 121; no language model involved), the Gemma 12B judge finds **90 %** of the statements supported and the NLI critic of Stage 9 accepts 85 % of the claims. The independent judge (Llama 3.1 8B) gave 0.60, but a hand check showed that it misreads results tables, so that figure is not evidence about the answers. The optional rewrite loop (off by default) raises the score of the judge it uses and nothing judge-free (numbers grounded 99 to 100 %, right numbers 8 to 9 of 9), and it doubles the time per question (10 to 21 s): **no benefit demonstrated**. | 30 questions, one run per set. No valid independent judge was found; the Gemma judge is of the writer's own family. |

**Honest reading for the paper.** The condition store works as a scope checker, and the one idea that carries the result is the joint-coverage check. The data do not show that the LLM agents or Stage 7 beat simple alternatives **on accuracy**, and claims should be worded to that. What Stage 7 does show, on 100 pairs and with a paired test, is that it raises about a third as many false "real contradiction" alarms as an LLM judge (7 % against 21 %) and names the condition that explains a difference more often than the LLM judge (61 % against 45 % on the held-out pairs, 50 % against 40 % on the first 50, with the two names that the labelling instructions give to "dev against test" counted as one); its weak point is that it gives up when the stored profiles lack a model size or a setting (section 4.5c). On 50 fresh pairs (section 4.5d) it again separated the three classes better than the baselines but its accuracy (66 %) was below the base rate (70 %), and a rule that always says EXPLAINED reaches 78 % or more on this kind of data and tells nothing apart. The mistakes turned out to be mistakes of the stored cards, not of the rules: with each card checked against its own table cell (section 4.5e) Stage 7 gets 88 % of 50 new pairs right (68 % before, p = 0.006), 82.5 % over all 200 labelled pairs, finds 8 of 11 not-comparable pairs where the baselines find none, and raises almost no false alarm. That is the first Stage 7 accuracy above the trivial rule with a reason behind it, on new pairs of this corpus (not on new table layouts).
On 11 October the stored profiles were corrected with their own tables by a code-only repair (section 4.1b): the share of completely right profiles rose from 37 to 61 % on 94 results of external-benchmark papers drawn after the code was frozen (strict 22 to 39 %) and the field F1 from 90.3 to 94.0, which is still **below the 85 % that was asked for** for completely right profiles; against the human gold of a public benchmark (4.1c) the pipeline finds 88 % of the results (a plain LLM with the same model 79.0 %) but agrees with the annotators' names for only 54.3 to 62.8 % of them. Stage 7 with the repaired store and policy B gets 88 % of 250 pairs right (the trivial rule 80 %), with 5 false GENUINE calls in 250 and none of the 3 real contradictions found.
Everything is measured against keys made by one AI labeller; swapping in human labels is a re-run of the same scripts (section 8).

## 1. What was tested

The architecture names four claims. Each is scored against its own key.

| # | Claim | Measured as | Key (size) | Compared with |
|---|---|---|---|---|
| 1 | The Condition Profile Extractor stores correct profiles | per-field precision / recall / F1 and the share of fully correct profiles | A: 320 random stored profiles from 10 papers (two sheets of 170 rows, 20 shared) | on the public MetaLead benchmark (section 4.1c): a plain-LLM extractor with the benchmark's own prompt; on the 320 profiles of key A: nothing |
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
first-pass labels of job C in Aditya's row order and parses to the same 50 labels. Tarun's pair sheet was filled by a different AI model, **Gemini 3.1 Pro** (web app, 8 Oct 2026). It got the text of the blank sheet, a written task (instructions and 5 pairs per message) and the PDFs of each batch, and never saw the first-pass labels or the system's verdicts.
It labelled all 50 pairs (the last batch had to wait until the tool's usage limit reset). That the PDFs were attached, in this round and in the held-out round, is the project lead's statement of 9 October 2026; the chats were not logged at the time. Its answers and the third look are in section 4.5b.

**Job C, held-out pairs (9 Oct).** After Stage 7 had been changed (section 6) a second set of 50 pairs was built so that the change could be tested on pairs it was not made from (`scripts/make_heldout_pairs.py`, seed 1016).
The mining and the sampling are those of the first set; the pool is the 232 conflicting pairs (of 365 mined) that share no pair of papers + subject and no stored profile with the first set, and the strata that steer the sample come from the *frozen* Stage 7 (every fix-A switch off), so the fixes did not choose their own test.
The 50 pairs come from 25 pairs of papers. The system's verdicts (frozen and current) went into a private file that was never printed or read before the labels were final. The first pass (Claude) was made blind to the system, from the sheet's excerpts and five PDF pages read as pictures.
The second pass was made by **Gemini 3.1 Pro** (web app; the model name and the attachment of the PDFs of each batch as stated by the project lead on 9 October, not logged at the time) from the blank sheet text, in 10 batches of 5 pairs with the same written instructions as for the first set. Details and results: section 4.5c.

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
18. **Measures added or redefined after seeing results (9 Oct), because the project lead asked for every metric to be above 80 %.** (a) Retrieval on the gold evidence pages and the end-to-end "evidence in the sources" and "out-of-corpus question ends with a warning" tests (section 4.7) were added because the earlier store-made retrieval numbers (language + task questions 60 % at the top 5, out-of-corpus questions flagged weak 70 %) were circular or looked at one mechanism of two; the old numbers are kept next to the new ones. (b) The condition-naming score with one name for "dev against test" (section 4.5c) was chosen after seeing the exact-match score of 25 %. Both are measurement choices made after the fact: they are labelled so, and neither changes a verdict or the system. What was **not** done: no metric was raised by changing the key, dropping hard items, or tuning on the test; Stage 7's accuracy (58 %) stays below 80 % and is reported as it is.
19. **Faithfulness and the rewrite loop (9-10 Oct), added at the project lead's request ("increase all metrics to 80 % or more"; they approved the loop as an off-by-default switch and one independent judge model).** (a) The design had replaced the original runtime loop by the NLI critic; the loop is now back as a switch, which is a departure from the architecture that stays off by default. (b) The RAGAS metrics are our own re-implementation (the `ragas` package is not installed): not comparable with published numbers. (c) The judge models and the loop settings (threshold 0.80, 3 rewrites, the Llama 3.1 8B judge) were fixed before the run and nothing was tuned on the results; the run was done once. (d) A first reading of the Llama judge's scores (0.60 to 0.82 with the loop) was **circular** and the cross-check with the other judge was added to avoid reporting it as a gain. (e) **Then a hand check (10 Oct) showed that the Llama judge itself is unreliable on results tables** (section 4.8, point 2), which reverses my first reading that the independent judge was the stricter and therefore better one: the 0.60 and the loop's apparent gain are mostly judge errors. Section 4.8 was rewritten; the Llama figures stay in the table as measured, labelled, and are not used as evidence. (f) A judge-free check (decimal numbers of each answer found in its sources) was added after seeing that the judges disagree; it is a proxy and cannot see a right number attached to the wrong model. No metric was raised by changing the key or by choosing the judge that gives the best figure: the Gemma figure is reported with its own-family caveat, the Llama figure with its failure.
20. **Changes made on 10 Oct after reading failures, at the project lead's request ("finish increasing accuracy today"); all are development choices.** (a) The Stage 5 reranker was swapped to bge-reranker-base and its threshold set to -3.0 after comparing three cross-encoders on the gold evidence pages and the store-made questions (section 4.7b); its end-to-end effect is not shown. (b) Stage 1 got a second reading, hyphen-proof language matching and extras read from the question, after one out-of-corpus question without a warning (section 4.7b). (c) **Stage 7 policy B (a condition recorded for only one paper is the probable explanation) was found by reading the mistakes of the first two pair sets, where it looked decisive (24 of 28 such pairs EXPLAINED, accuracy 58 to 74 and 82 %). A fresh set of 50 pairs was built and labelled blind to the system by one AI labeller (the system's verdicts for it were never printed; the baselines were run after the labels were final) and scored once: policy B there equals "always EXPLAINED" (70 %) and loses the NOT COMPARABLE class (macro-F1 57 to 41), so it is OFF by default and the frozen Stage 7 stays the headline** (section 4.5d). The label errors that a single labeller can make are the soft spot of that test (9 of its 50 labels are medium confidence; F01, F32 and F44 are judgement calls between EXPLAINED and a genuine difference between leaderboard snapshots). (d) What was not done: no metric was raised by changing a key, dropping hard pairs, or choosing the variant that scores best on the data it was found on; the headline Stage 7 numbers are still the frozen ones, and "accuracy of 80 % or more" for Stage 7 is reached only by a rule that always says EXPLAINED.
21. **Table grounding (10 Oct, evening), a method change for Stage 7 at the project lead's request ("do whatever is necessary to increase the accuracy of even Stage 7").** (a) The design was made on three labelled pair sets whose mistakes I had read; their numbers are development numbers. (b) Before the fourth set was built the code and the tests were frozen; no rule was changed after its labels existed. (c) The fourth set could not be made of unused profiles (346 of the 365 mined pairs share a profile with an earlier set): a profile may appear in a new pair, no pair repeats; this is a weaker independence than for the first three sets and is stated in 4.5e. (d) Its labels were made with the stored cards and the raw chunk rows in view (not the excerpts of the label sheets, which mislead where a value occurs twice in a table) and blind to the system's verdicts. (e) While reading the stored cards I found that one label of the earlier fresh set (F46) is probably wrong (the stored card is a legitimate WNLI figure; I had labelled it from an excerpt as an STS-B misread): it was NOT changed after seeing the system's verdicts, and it is left as labelled. (f) A fix to the table parser (a heading cut at column borders was read as a result row) is part of the change; the stored profiles were not re-extracted, so the stored settings and sizes are still the old ones and Stage 7 corrects them at query time. (g) The earlier statement in this document and in `CLAUDE.md` that Stage 7 accuracy above 80 % is reachable only by a rule that always says EXPLAINED was true of policy B and is no longer true of the system: with the cards read again it is reached, with the reason shown in 4.5e.
22. **The profile repair (10-11 Oct), at the project lead's order "all metrics at least 85 %".** (a) The rules were written from the 320 hand-checked profiles and a first blind sample (E1, 120 profiles); E1 became a development set when rules were written from its own errors (completely right 60 % at the first look, 77 % afterwards). (b) Applying the repair to the whole store and reading its most frequent changes (a check the 120-profile sample could not give, because the harmful patterns are rarer than 1 in 100) found harm, which was removed before the clean test: `model_row` replaced the stored system by the row label where that label is a category or a hyper-parameter ("Other", "MAX", "Multi"); `model_name` shortened composite names ("BiLSTM+ELMo" to "+ELMo") and turned "mT5-XL" into "mT5-XXL" through a subsequence match; `dataset_task` replaced a category ("Coref", "MASSIVE (Intent)") by its benchmark and lost the category; the column names "Te" and "De" were read as the languages Telugu and German, a ToxiGen group "Chinese" as a language, and "mr" (Marathi) matched the review data set "MR". (c) One label of E1 (E016) was wrong and was corrected after the better table reader showed the cell is Telugu, not Malayalam; the correction lowers the stored and the first repaired version by one profile each (47 to 46 and 71 to 70 of 119) and is in the development row of the table in 4.1b. (d) The labeller of E1 and E2 is the AI that wrote the rules; the A / B order was random but partly guessable. (e) E2 was labelled with a stricter policy than the repair's own (a split counts only if the table, its caption or the sentence about the table states it). (f) The code hashes at the moment E2 was drawn are recorded (`data/labelling/private/extraction_test2/code_hashes.txt`); re-running the repair on the 100 items with the final code reproduces the stored 'repaired' versions exactly (checked on 11 Oct), so nothing that the repair does changed after E2 was labelled.
23. **External benchmark and the plain-LLM baseline (10-11 Oct).** (a) MetaLead was read with the frozen extractor, so the stored numbers are clean; the repaired numbers are development numbers because the rules that read the paper's own text (abbreviations that the paper defines, a data set named in the caption; at least two of them) were designed after reading this benchmark's disagreements. (b) The scorer is mine and looser than the benchmark's own; the metric aliases and the task families were written before any score was computed, the surface-form normalisation (years, "v5" = "5.0", an acronym against its words, a stem of six letters) after seeing that spelling variants were counted as errors; all systems are scored with the same matcher. (c) The baseline's first pass lost 23 of 464 windows to the output limit (invalid JSON); it was re-run with window splitting before it was reported (the first-pass numbers are kept in `eval/labels/metalead_eval_baseline_v1_nosplit.json`). (d) No result of the benchmark's authors was re-run.
24. **Policy B adopted as a default (11 Oct), a departure from 10 Oct.** On 10 Oct policy B was left OFF because it equalled "always EXPLAINED" (item 20). With grounding it does not: fresh-2 47 against 44 of 50 (a post-hoc reading of a test set), then a confirmation on a new set (fresh-3, section 4.5f): 46 against 42, "always EXPLAINED" 45. Because the first look at fresh-2 was post-hoc the clean evidence for the switch is fresh-3 alone (gained 5, lost 1, p = 0.22) plus the pooled reading (8 / 1, p = 0.039); the switch can be turned off (`[contradiction] one_sided_conditions_explain`) and the numbers without it are in the same table.
25. **Three fixes after the first system run on the repaired store (11 Oct), section 4.7c.** (a) Stage 1 kept no N-shot setting (`setting_tags` has no tag for "5-shot", so `clean_conditions` deleted it) and a task that is only a setting word ("zero-shot") stayed in the task field; (b) Stage 6's joint check did not include a named setting, so "5-shot" was covered by any passage that mentions 5-shot (it now includes a regime setting, not a dev / test split); (c) retrieval cards did not name the data set version. All three are general, were found by reading single questions (B-AI-16, B-AI-18, B-AI-01) and were applied before the final run; the gold questions are therefore development material for them (not held-out). The first system run on the repaired store was kept in the files (`*.oct11.*`) and is reported next to the final one; the API-part evaluations (sources, out-of-corpus) were only run after the fixes. (d) Two further changes came from a probe of ten questions outside the gold set (4.7c): the joint check was first made for every setting and warned falsely on a dev / test split that the store did not record, so it was restricted to regime settings; and a recorded "Cross-lingual Transfer" setting now covers a requested "zero-shot" in Stage 6's matching (`setting_tags`, which Stage 7 compares, is unchanged: a replay of all 250 pairs changed no verdict). The whole evaluation pass was repeated after (d) (`*.oct11f.*`) and gave the numbers of 4.7c without any difference.
26. **Why two of the "one missing" questions had a right warning on 10 Oct.** B-AI-16 (5-shot HellaSwag) was warned because the stored data set was "ellaSwag" and B-AI-18 (zero-shot MNLI) because Stage 1 had filed "zero-shot" as a task: both warnings were correct in the key's sense (a warning was due) but named a wrong condition (the 10 Oct count of 17 of 19 right conditions named already showed it). With the repaired names and the fixes of item 25 they are warned for the right reason (condition named: 19 of 19).

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
This is one labeller twice (section 3, item 9), so it shows that the labelling is stable, not that two people agree. There is **no baseline on these 320 profiles**; a plain-LLM baseline with the benchmark's own prompt exists on a public benchmark (4.1c), and what a code-only repair does to these errors is in 4.1b.

### 4.1b The profile repair: stored profiles corrected with their own tables (10-11 Oct)

**Verdict: a code-only repair (no language model; the raw rows are never rewritten) raises the share of completely right profiles on 94 results of 34 papers of an
external benchmark, drawn after the code was frozen, from 37.2 % to 60.6 % (23 profiles fixed, 1 broken, sign test p < 0.0001; 95 % CI of the repaired share
50.5-69.9) and the field-level micro-F1 from 90.3 to 94.0 (strict reading: 22.3 to 39.4 %, F1 84.9 to 90.0). Almost all of it is the setting field
(F1 58.4 to 87.1). It does not reach 85 % completely right profiles: what is left is mostly a data set name that is a header fragment or a column label ("IS Data" for
ATIS, "English" for CoNLL-2003; the real name is in the paper's text), a number read from another block of a stacked table, and settings that the table never
states.**

*Why.* The hand check of 320 stored profiles (section 4.1) found the numbers right (value F1 99.7 %) and the names weak (lenient F1: task 78.0, model 85.2, data set
87.0, setting 88.2 / 72.2 strict; 157 of 311 real results, 50.5 %, completely right). The errors repeat in a few patterns and each pattern is settled by the table
the number came from (or by a definition the paper itself writes), so a repair can be pure code and every correction can be traced to the text that states it.

*What it does* (`src/cgrag/ingestion/repair.py`, 25 named rules, each switchable; `src/cgrag/knowledge/` holds the small lists it uses):
(1) **task** from the benchmark (MRPC is paraphrase detection, not sentiment analysis; "EnDe" is machine translation; a training regime or a filler word such as
"multi-task" is not a task) unless the caption itself calls it so; (2) **model**: a name printed vertically beside a group of rows arrives in pieces ("L" +
"lama 1") and is joined by the table reader, an anagram or fragment of a system name that the paper writes ("aLLMA", "coFaln", "Fln") is matched to it, a
configuration label ("ALBERT: E = 64", "K = 2^18") becomes the setting and the paper's own system the model; (3) **setting**: a split that no table, caption or
sentence about the table states is removed, a split that the block heading or the column states is added, "single model" is removed when nothing says it, and a
training regime is read from the block heading ("Translate-train", "Cross-lingual zero-shot transfer"; stray glyphs of the heading are removed before the words are
compared); (4) **language** from the cell's own column when the table is a table of languages (the second half of a table printed in two halves) and none for an
"avg" column or a sentence such as "Results averaged across languages"; (5) **data set**: the split prefix ("Dev Set SST-2"), a name cut at a column border
("SQuA"), a name expanded from the paper's own definition ("WN16" = WNUT-16), the benchmark that a set of language columns identifies when a caption was lost
(XQuAD, MLQA, TyDi QA, XNLI, PAWS-X), a data set named in the caption; (6) metric abbreviations that the paper defines ("scc" = Spearman correlation
coefficient), and a model size that is not a size is removed or reduced to its size word ("DeBERTa V3 Large" -> large). **Table reader**: a header row that is
repeated in the middle of a wide table (two halves printed one above the other) now switches the column names; before, the second half's values got the first
half's language names.

*How it was made, and what is development.* Rules were written from the 320 hand-checked profiles and from a first blind sample (E1: 120 stored profiles of the 28
papers, labelled field by field with the stored and the repaired version in random order, A / B). On E1 the first repair gave 39.5 to 59.7 % completely right
(26 fixed, 2 broken; strict 22.7 to 37.0 %); rules written afterwards from E1's own errors reach 38.7 to 77.3 % (strict 21.8 to 54.6 %, micro-F1 90.8 to 97.3), which is a
**development number**: E1 is no longer a test. Applying the repair to the whole store and reading its most frequent changes found rules that did harm on
rarer patterns (section 3, item 22); they were removed or tightened before the clean test. The code was then frozen (hashes recorded) and a **clean sample (E2)**
was drawn: 100 random profiles of the benchmark of 4.1c (they fall in 34 of its 43 papers: NER, summarization, slot filling, translation; tasks and table layouts the
rules were not written for), labelled the same way. One exposure has to be named: four rules that read the paper's own text (`dataset_paper_abbrev`,
`dataset_caption`, `metric_paper_abbrev`, `task_paper`) were designed after reading disagreements on the same benchmark's papers (4.1c); with those four switched off
(they change 11 of the 100 items) E2 gives 58 of 94 completely right (61.7 % lenient, F1 94.1) and 37 of 94 (39.4 % strict, F1 89.3), so the clean gain does not
depend on them; they matter for agreement with the annotators' names in 4.1c, not for the correctness of the field.

| Sample | results | version | completely right (lenient) | completely right (strict) | field micro-F1 lenient / strict |
|---|---|---|---|---|---|
| E1, first look (28 papers; the first repair; before the label correction of item 22c, which gives 38.7 -> 58.8 %) | 119 | stored -> repaired | 39.5 -> 59.7 % (26 / 2, p < 0.0001) | 22.7 -> 37.0 % (18 / 1) | 91.6 -> 94.7 / 86.7 -> 90.4 |
| E1 after rules were written from it (**development**) | 119 | stored -> repaired | 38.7 -> 77.3 % | 21.8 -> 54.6 % | 90.8 -> 97.3 / 86.0 -> 94.0 |
| **E2, clean (benchmark papers, 34 of 43)** | 94 | stored -> repaired | **37.2 -> 60.6 %** (23 / 1, p < 0.0001) | **22.3 -> 39.4 %** (16 / 0, p = 0.0001) | **90.3 -> 94.0 / 84.9 -> 90.0** |

Per field on E2 (lenient F1, stored -> repaired): setting 58.4 -> 87.1, model_size 96.3 -> 99.5, task 90.4 -> 91.5; unchanged: dataset 83.0,
model 92.6, metric 93.6, language 98.4, value 100. Of the 37 profiles that are still not completely right (lenient), the wrong or missing field is the data set in 16, the setting in 13, the task in 8, the model in 7, the metric in 6, the language in 2 and the model size in 1 (a profile can have more than one). Six of the 100 profiles were not results (a figure axis, a speed-up of the
reference system itself, a case-study example).

*What it does to the system.* In the 28-paper store 5,383 of 10,275 profiles change (52.4 %): setting 3,321 fields, task 2,119, data set 915, model 467, language
433, model size 64. Stage 7's shipped configuration is unchanged by it (165 against 164 of 200 pairs; fresh-2 44 against 44), because table grounding already
corrected the cards at query time; the variants without grounding gain (frozen rules 122 to 137, fix A 131 to 144 of 200): the repair does in the store what
grounding does at query time. The system switch is `[features] profile_repair` (on); the raw profiles stay in `profiles`, the corrections in `profile_repairs`.

*Limits.* The labeller of E1 and E2 is the AI that wrote the rules: the order A / B was random, but an expanded abbreviation or a removed fragment shows which
version is which, so the blinding is partial. E2 was labelled with a stricter policy than the repair itself uses (a "test set" is right only where the table, its
caption or the sentence about it states it; the repair also accepts the paper's general statement), which lowers both versions. The share of completely right
profiles (60.6 % lenient, 39.4 % strict) is **below 85 %**; the field-level F1 (94.0 / 90.0) is above it. A data set group or subset (an MMLU subject, a BOLD group,
a RACE subset) is still not recorded in any field (strict reading).

### 4.1c An external check on a public, human-made benchmark, and the plain-LLM baseline (MetaLead, 10-11 Oct)

**Verdict: the pipeline finds 88.4 % of the 3,568 results that the human annotators of MetaLead wrote down (the plain LLM with the same model and MetaLead's own
prompt finds 79.0 %), and among the results it finds it names the data set, metric and task like the annotators for 54.3 % (repaired:
62.8 %; plain LLM: 38.3 %).**

*Benchmark.* MetaLead (Timmer, Bölücü, Wan, EACL 2026): 43 NLP papers with 23 tasks (most results: named entity recognition 1,016, summarization
923, intent detection and slot filling 310, machine translation 198, word sense induction 144), 3,568 human-annotated results with task, train and test data set,
metric and score. The extractor, its prompts
(1 October) and the model (gemma4:12b) were not changed for this benchmark, which was downloaded on 10 October; the stored numbers are therefore the pipeline as it
was (the table reader got one extension, repeated header rows, after the papers had been read; the benchmark's stored profiles were not re-extracted). The papers
were read into a separate index (4,343 profiles). The scorer (`scripts/eval_metalead.py`) is **not** MetaLead's own (a set-based exact match of the 6-tuples <task, train data
set, test data set, metric, score, experiment type> after a language model had normalised each predicted name to the paper's gold names; the best published F1 is 48.9 in
the closed-domain and 30.0 in the open-domain setting, Gemini-2.5-Pro and o4-mini); it is looser and deterministic: *coverage* = a prediction of the same paper has
the result's number (tolerance 0.005, also x100); *agreement* = among the covered results, the best prediction with that number has a data set / metric / task
name that agrees (normalised token sets contain one another, language codes expanded, years and versions normalised, an acronym matches its words, metric aliases,
task families); *precision* = share of predictions whose number is in the paper's gold. The numbers are therefore not comparable with the published ones.

*Plain-LLM baseline.* The same model, MetaLead's own extraction prompt, the raw text of each paper (PyMuPDF, reference list cut off) in windows of 6,000
characters (the model is served with an 8k context; a window whose answer was cut off is split in two and asked again), no table recognition, no cards, no repair
(`scripts/baseline_llm_extract.py`). A first pass without the splitting lost 23 of 464 windows (coverage 51.9 %, all three 48.6 %); the figures below are
the second pass, in which 49 windows were split and 9 of the resulting pieces still could not be parsed and were dropped.

| System | papers | coverage | data set | metric | task | all three agree | end to end (covered and all three) | precision |
|---|---|---|---|---|---|---|---|---|
| Plain LLM (gemma4:12b, MetaLead prompt, text windows) | 43 | 79.0 | 60.7 | 78.4 | 67.9 | 38.3 | 30.3 | 79.7 |
| **Pipeline, stored profiles** (clean) | 43 | **88.4** | 67.9 | 90.0 | 84.6 | 54.3 | 48.0 | 79.7 |
| Pipeline + repair (**development**: rules that read the paper's own text were designed after reading this benchmark's disagreements) | 43 | 88.4 | 73.3 | 91.9 | 86.3 | 62.8 | 55.5 | 79.7 |

*Reading.* (1) The pipeline's advantage is in **finding** results: table recognition and the per-table views let it read the numbers of dense tables that a text window
loses (88.4 against 79.0 %; macro over papers 89.9 against 86.3 %), and its end-to-end share (result found and all three names
agree) is 48.0 against 30.3 %. Counted per paper the pipeline finds more of a paper's results than the plain LLM in 24 papers and fewer in 11
(sign test p = 0.04); for the end-to-end share the papers split 21 to 17 (p = 0.63; repaired 24 to 15, p = 0.20), so the pooled end-to-end
difference is carried by a few large papers and is **not significant paper by paper**. (2) The names are the common weak point of both: among the results each system finds, the data set names agree for
60.7 % (plain LLM), 67.9 % (pipeline, stored) and 73.3 % (repaired). The data set is often not in the table at all (the column is "POS" or
"Chunking", the data set is in a sentence of the paper), the annotators' normalised names ("CoNLL-2003 - English") are not the paper's ("English NER"), and an
abbreviation is defined in the text ("ON5E"). The repair's paper-grounded rules (abbreviations the paper defines, data sets named in the caption, the paper's main
task for a missing task) lift the pipeline's three names from 54.3 to 62.8 %; they do not close the gap. (3) Coverage (88.4 %), metric agreement
(91.9 %) and task agreement (86.3 %) are at or above 85 %; the data set agreement (73.3 %) and the three names together (62.8 %) are **below 85 %**.

*Limits.* One benchmark of 43 papers from NLP tasks the 28 development papers do not cover, one run per system. The agreement scorer is lenient and my own, and the
benchmark's gold names were read while repair rules that read the paper's own text were designed (item 23 of section 3), which is why the repaired row is a development number. The
reference results of the benchmark's authors (GPT-4.1, Gemini-2.5-Pro, Llama 3.3 70B) were not re-run: no external API is part of the system, and a 70B model does not
fit the machine.

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
**Condition naming with one name for dev / test.** The labelling instructions list "dev against test" under two names ("dataset version / split" and "setting"), so an exact match punishes a labeller or the system for choosing the other name. Counted as one name (`condition_attribution_merged` in the scorer; a reading chosen after seeing the results, so it is not held-out), Stage 7 names the right condition for **61 %** of the 28 held-out pairs (mean overlap 74 %) and Gemma for 45 % of 31 (58 %); on the first 50 pairs Stage 7 gets 50 % of 26 (overlap 71 %) and Gemma 40 % of 30 (53 %). Stage 7 names the condition better than the LLM judge on both sets (the judge is asked for the differing conditions too), which is the second clear difference between them besides the false alarms; the samples are small (28 and 26 pairs) and the merged reading was chosen after the fact.

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

### 4.5d Stage 7 on a fresh set of pairs, and the "one-sided condition" policy (10 Oct)

**Verdict: on 50 new pairs that no rule was tuned on, Stage 7 gets 66 % of the verdicts right (the share of EXPLAINED pairs, which "always EXPLAINED" scores, is 70 %); but it separates the three classes better than the baselines (macro-F1 57 against 41 for the Gemma judge and for "always EXPLAINED"), raises no false "real contradiction" alarm (0 of 50, the Gemma judge 6 of 50) and finds 5 of the 15 not-comparable pairs (both baselines 0 of 15). A policy that I found by reading the mistakes of the first two sets, "a condition recorded for only one paper is the probable explanation", lifts accuracy on the fresh set to 70 % only by calling every pair EXPLAINED; it is therefore OFF by default.**

*Why a fresh set.* The first 50 pairs were used to design fix A, and the 50 held-out pairs were scored once for the frozen commit, but I read their mistakes afterwards. The pattern that came out of that reading (see below) is a design idea found on those two sets, so they cannot test it. 50 new pairs were built with the same sampler (`scripts/make_heldout_pairs.py --prefix F --name fresh`; no profile, paper pair or subject of the first two sets reused; strata from the FROZEN rules), labelled by one AI labeller (Claude) blind to the system's verdicts (the key file was never printed), with the PDF pages rendered as pictures wherever the excerpt hides a block heading (the same trap as in section 3, item 15: ALBERT's Table 9 and XLNet's Table 5 hide "Ensembles on test (from leaderboard)"), and scored once. **There is no second labeller for this set** (no kappa); the gold has 35 EXPLAINED, 15 NOT COMPARABLE (every one with an extraction error flagged: a row of RoBERTa or XLNet-Base stored as "BERT", an F1 stored as an accuracy, STS-B stored as WNLI, a margin or a subset read as a score), no GENUINE pair, 9 labels of medium confidence.

*The pattern.* On the 28 pairs of the first two sets where the frozen rule says "model size (or setting, or dataset version) is recorded for only one of the two papers, so it cannot be shown that the conditions are the same" (NOT COMPARABLE), the key says EXPLAINED 24 times and NOT COMPARABLE 4 times. Policy B (`[contradiction] one_sided_conditions_explain`) turns this rule into "EXPLAINED: probable explanation, naming that condition". On the first two sets it lifts accuracy from 58 to 74 % (first) and from 58 to 82 % (held-out).

| Set (gold) | Frozen Stage 7 | Fix A (shipped) | Fix A + policy B | Gemma 12B judge | Always EXPLAINED |
|---|---|---|---|---|---|
| First 50 (41 expl., 9 n.c.) *development* | 29 (58 %), macro-F1 50 | 35 (70 %), 68 | 45 (90 %), 91 | 30 (60 %), 41 | 41 (82 %), 45 |
| Held-out 50 (41 / 7 / 2 genuine) *development for B* | 29 (58 %), 27 | 29 (58 %), 27 | 41 (82 %), 30 | 33 (66 %), 39 | 41 (82 %), 30 |
| **Fresh 50 (35 expl., 15 n.c.) clean test** | **33 (66 %), 57** | **33 (66 %), 57** | 35 (70 %), 41 | 32 (64 %), 41 | 35 (70 %), 41 |
| All 150 pairs (not-comparable pairs found) | 91 (61 %), 33 (9 of 31) | 97 (65 %), 38 (14 of 31) | 121 (81 %), 43 (8 of 31) | 95 (63 %) | 117 (78 %) |

(Each cell: pairs right (share), macro-F1. The fresh row is the only clean test of anything made after the first 50 pairs.)

Reading:
1. **Fix A does not generalise.** It lifts the first set (the one it was tuned on) from 58 to 70 % and changes no verdict on the held-out set or on the fresh set: 66 % and 58 % stay what they were.
2. **Policy B is the base-rate classifier.** On the held-out set (82 %) and the fresh set (70 %) it makes every one of the 50 pairs EXPLAINED, which is exactly what "always EXPLAINED" does; its accuracy equals the share of EXPLAINED pairs. On the fresh set the 12 pairs it changes are 7 gains and 5 losses (net +2), the not-comparable pairs it used to find (5 of 15) are all lost, and macro-F1 falls from 57 to 41. The pattern that looked so strong on the first two sets (24 of 28) is the base rate: on the fresh set only 7 of the 12 such pairs are EXPLAINED, and over all 40 it is 31 of 40 (78 %), the same as the share of EXPLAINED pairs overall (78 %). Accuracy above 80 % is therefore reachable by a rule that always says EXPLAINED, and by nothing that tells the classes apart. **Policy B stays a switch, off.**
3. **What the clean test supports.** The frozen rules separate the classes better than the two baselines (macro-F1 57 against 41 and 41), never raise a false alarm on the fresh set (Gemma: 6 of 50, 12 %) and name the condition more often than the Gemma judge (among the pairs that both the key and the system call EXPLAINED, the right condition is named in 68 % of 28 for Stage 7 and 38 % of 32 for Gemma); over all 150 pairs the frozen Stage 7 calls a pair a real contradiction wrongly 7 times of 148 (5 %; fix A: 3) where the Gemma judge does it 28 times (19 %). It does **not** beat "always EXPLAINED" on accuracy (66 % against 70 %), and it found none of the 2 real contradictions in the corpus (only the held-out set has any).
4. **Where the not-comparable class is lost.** The 15 NOT COMPARABLE pairs of the fresh set are extraction errors, not missing conditions: a result row of another model stored under a BERT name, an F1 stored as an accuracy, one column stored as another, a margin or a subset stored as a score. A rule on the stored conditions cannot see those; checking a profile against its source row (is the stored model the row's label? is the number in the stored column?) is the lever, and it is an extraction task (section 7).

Limits: one labeller (the fresh gold has no agreement score, and the same AI helped build the system); 50 pairs (95 % CI of 66 % is 52-78 %); the fresh pairs are drawn from the same 28 papers, mostly the GLUE-table family (BERT, RoBERTa, XLNet, ALBERT, ELECTRA, DeBERTa, TinyBERT), so the same few tables occur in many pairs (ALBERT's Table 9 page is one side of 12 of the 50 pairs); there is no GENUINE pair in it, so it says nothing about finding real contradictions. Scripts: `scripts/make_heldout_pairs.py`, `scripts/fill_job_c_from_ai.py`, `scripts/run_pair_baselines.py`, `scripts/score_stage7_variants.py` (replays `classify` on the stored profiles of all 150 pairs and reproduces the stored verdicts of the frozen commit pair by pair); result `eval/labels/stage7_variants.json` (aggregates only).

### 4.5e Stage 7 with table grounding: the cards were the problem (10 Oct, evening)

**Verdict: on 50 new pairs that were built and labelled after the design was frozen, Stage 7 with table grounding gets 44 of 50 right (88 %), against 34 (68 %) without it (11 gained, 1 lost, sign test p = 0.006), 34 (68 %) for the Gemma 12B judge and 39 (78 %) for "always EXPLAINED" (8 gained, 3 lost against that rule, p = 0.23). It tells the classes apart far better than either baseline (macro-F1 83 against 60 before, 42 for Gemma and 44 for "always EXPLAINED"), finds 8 of the 11 not-comparable pairs (both baselines 0 of 11) and raises no false "real contradiction" alarm (Gemma 6 of 50). Over all 200 pairs of the four sets it is right 165 times (82.5 %), against 131 (65.5 %) before. It is on by default (`[contradiction] table_grounding`).**

*What was wrong.* Not the rules of 4.5d but the stored cards that they compare. The block heading of a table ("Ensembles on test (from leaderboard as of Sept. 16, 2019)") never reached the rows under it: the layout cuts a heading at column borders, its last piece ("019)") looks like a number, so the parser read the heading as a result row and four ensemble rows of ALBERT's GLUE table kept the heading of the dev block ("single-task single-model"); a split ("dev" / "test") was lost from other settings; a size field held "ensemble", "1M steps" or "1.75M"; a row of one system was stored under another's name (RoBERTa's XLNetLARGE block stored as BERT-large) and a column under another data set (STS-B stored as WNLI). Two numbers that a reader sees at once as dev against test on one side, ensemble on the other, looked "the same" or "recorded for one paper only" to Stage 7.

*What grounding does* (`src/cgrag/pipeline/grounding.py`, pure code, no language model; the table parser is `src/cgrag/ingestion/tables.py`, whose cut-heading bug is fixed too):
1. it finds the table cell of a stored result in its own chunk: cells that carry the card's number, best fit to the card's model (row label) and data set (column name), the split of the stored setting deciding between equal numbers; no guess when nothing fits;
2. it reads what the table states: the block heading gives the split (dev / test, a leaderboard is test) and the system kind (single / ensemble); the caption gives the split where no heading does, the protocol ("logistic regression ... using the sentence embeddings as features" = frozen embeddings) and a size for every row ("results for large models", "a 24-layer architecture"); the row label gives a size (BERT-large, RoBERTa_base, BERTTINY); a stored size that is not a size is ignored;
3. it calls a card suspect when the table contradicts it: the model does not fit the row label (or, for a variant row such as "+ additional data", the caption's subject), the data set is another benchmark task than the column (only task columns count: a language or metric column says nothing), or the number is a part inside a bracket of its cell ("83.2 (86.5/81.3)": RACE Middle against the total);
4. Stage 7 compares the enriched cards (a suspect card gives NOT COMPARABLE with the cell it did not fit named in the reason).
A condition is filled only when the table states it. The stored profiles are not changed (the store was not re-extracted); the correction happens at query time.

*Protocol.* The design was made on three labelled sets (the first 50 pairs, the 50 held-out pairs, the 50 fresh pairs of 4.5d: their error types were read while the rules were built, so they are development sets for this change), then **frozen** (code, 240 tests), and then a fourth set was built (`make_heldout_pairs.py --prefix G --name fresh2 --allow-profile-reuse --reuse-subjects`; 346 of the 365 mined conflicting pairs share a profile with an earlier set, so a profile may appear in a new pair, no pair of two profiles repeats), labelled by one AI labeller with the stored card fields and the raw chunk rows in front of it and blind to the system's verdicts (the verdict fields of its key were never printed), and scored once, the baselines having been run after the labels were final. The labelling dump showed the stored cards because the sheets' excerpt of earlier sets had misled (it highlights the first cell that carries the number, not the card's cell).

| Pairs right of 50 (macro-F1) | Frozen | Fix A, before grounding | **Fix A + table grounding** | Gemma 12B judge | Always EXPLAINED |
|---|---|---|---|---|---|
| First 50 (41 expl., 9 n.c.) *development* | 29 (58 %), 50 | 35 (70 %), 68 | **41 (82 %), 80**; +6 / -0, p = 0.03 | 30 (60 %), 41 | 41 (82 %), 45 |
| Held-out 50 (41 / 7 / 2 genuine) *development* | 29 (58 %), 27 | 29 (58 %), 27 | **39 (78 %), 35**; +10 / -0, p = 0.002 | 33 (66 %), 39 | 41 (82 %), 30 |
| Fresh 50 (35 / 15) *development* | 33 (66 %), 57 | 33 (66 %), 57 | **41 (82 %), 78**; +9 / -1, p = 0.02 | 32 (64 %), 41 | 35 (70 %), 41 |
| **Fresh-2 50 (39 / 11): the clean test** | 31 (62 %), 49 | 34 (68 %), 60 | **44 (88 %), 83**; +11 / -1, p = 0.006 | 34 (68 %), 42 | 39 (78 %), 44 |
| All 200 (not-comparable pairs found) | 122 (61 %), 33 (12 of 42) | 131 (66 %), 38 (20 of 42) | **165 (82.5 %), 51 (26 of 42)** | about 129 (64 %) | 156 (78 %) |

False "real contradiction" calls over the 200 pairs (196 of them are not real contradictions): grounding 4 (2 %), the Gemma judge 33 (17 %). Condition named correctly (both the key and the system say EXPLAINED, merged naming): 44 % of 36 pairs on the clean set for Stage 7 against 26 % of 34 for Gemma.

Reading:
1. **The gain is real and it comes from the cards.** On all four sets grounding gains pairs and loses at most one (+36 / -2 against fix A over the four sets, each set significant by a sign test). The gains are pairs whose dev / test / ensemble difference was in the table but not on the card; the losses are cards that grounding cannot read.
2. **Against the trivial rule the margin is smaller in accuracy than in everything else.** 88 % against 78 % on the clean set is not significant (p = 0.23): 78 % of the pairs are EXPLAINED, so "always EXPLAINED" already scores that. What grounding adds is the three-class skill: 8 of 11 not-comparable pairs found, 83 against 44 in macro-F1, no false alarm. Accuracy over the four sets, 82.5 %, is the first value that exceeds the trivial rule (78 %) with an actual reason behind it.
3. **Fix A did not generalise, grounding does.** Fix A changes nothing on the held-out and the fresh set (4.5d); grounding gains on each of the three sets it was not tuned to individually, and then on the clean set.
4. **What is left.** The not-comparable pairs that grounding misses are metric mixes (an F1 stored as an accuracy: TinyBERT's test F1 against a dev accuracy, G09 / G43), a margin or a GLUE score read as a result when the number is in prose, and human-performance rows that fix A already catches; 6 of the 11 not-comparable pairs of the clean set are human rows (fix A, 4.5), so grounding's own share is 2 of the other 5 (the other 3 are the metric mixes and the GLUE score read as MNLI). The one pair whose labels I now doubt (F46, labelled NOT COMPARABLE from an excerpt; the stored card is a legitimate WNLI ensemble figure) was left as labelled.

Limits: one labeller (no agreement score) and the same AI built the system; the clean set reuses profiles from earlier sets in new pairs and comes from the same 28 papers, mostly GLUE and SQuAD tables repeated across papers, so it shows that grounding generalises to **new pairs of this corpus, not to new table layouts**; the clean set has no genuine contradiction (neither of the 2 real contradictions of the held-out set is found, as before); 9 of its 50 labels are medium confidence. Scripts: `scripts/score_stage7_variants.py` (replays `classify` with and without grounding on all 200 pairs, paired sign tests; reproduces the stored verdicts of the frozen commit pair by pair), `scripts/make_heldout_pairs.py`, `scripts/fill_job_c_from_ai.py`, `scripts/run_pair_baselines.py`; tests `tests/test_grounding.py` (10); result `eval/labels/stage7_variants.json` (aggregates only).

### 4.5f Stage 7 after the profile repair, a third fresh set, and policy B with grounding (11 Oct)

**Verdict: the repaired store leaves the 10 Oct configuration of Stage 7 where it was (165 of 200 pairs right on the raw store, 164 on the repaired one). Policy B, which
alone only reproduced "always EXPLAINED", is a small gain once the cards are grounded: on a third fresh set of 50 pairs (J01-J50, mined from the repaired store, labelled
blind to the system after the idea had come from fresh-2) grounding + policy B is right on 46 pairs (92 %, CI 81-97), grounding alone on 42 (84 %), "always EXPLAINED" on
45 (90 %; the set has 45 EXPLAINED, 4 NOT COMPARABLE and 1 GENUINE pair) and the Gemma 12B judge on 31 (62 %); macro-F1 51 against 46, 32 and 33. Over all 250 pairs
(five sets, four of them used in development) 88.0 % (CI 83-92) against 82.4 % for grounding alone and 80.4 % for "always EXPLAINED". The clean evidence for B is
small (fresh-3 alone: 5 pairs gained, 1 lost, p = 0.22), so B is ON by default from 11 Oct as a flagged departure (`[contradiction] one_sided_conditions_explain`) and
can be switched off; it costs some not-comparable recall (25 of 46 such pairs found against 29).**

| Pairs right of 50 (macro-F1), repaired store | frozen | fix A | grounding (10 Oct) | **grounding + policy B (shipped from 11 Oct)** | always EXPLAINED | Gemma 12B judge |
|---|---|---|---|---|---|---|
| first 50 (development) | 33 (59) | 37 (72) | 40 (78) | **46 (92)** | 41 (45) | 30 (41) |
| held-out 50 (development) | 36 (33) | 36 (33) | 39 (36) | **41 (30)** | 41 (30) | 33 (39) |
| fresh 50 (development for grounding) | 32 (56) | 32 (56) | 41 (78) | **40 (73)** | 35 (41) | 32 (41) |
| fresh-2 50 (the clean test of grounding; policy B proposed after seeing it) | 36 (56) | 39 (69) | 44 (83) | **47 (90)** | 39 (44) | 34 (42) |
| **fresh-3 50 (clean, 11 Oct)** | 40 (44) | 40 (44) | 42 (46) | **46 (51)** | 45 (32) | 31 (33) |
| all 250 | 177 | 184 | 206 (82.4 %) | **220 (88.0 %)** | 201 (80.4 %) | 160 (64 %) |

*Protocol.* Policy B was found on the first two sets, was OFF on a fresh set (there, without grounding, it equalled "always EXPLAINED") and was OFF when grounding was
designed. With grounding it looked different on fresh-2 (47 against 44 of 50, 3 gained and 0 lost), which is a post-hoc reading of a set that had served as a test.
To confirm it without reusing a test, fresh-3 was made from the repaired store (`make_heldout_pairs.py --seed 1019 --prefix J --name fresh3 --allow-profile-reuse
--reuse-subjects`, the four earlier keys excluded) and labelled by one AI labeller (this assistant) from the table text and captions (no PDFs attached), blind to
the system's verdicts; the baselines were run after the labels were final. Paired sign tests, grounding + B against grounding alone: fresh-3 gained 5, lost 1
(p = 0.22); fresh-2 and fresh-3 together gained 8, lost 1 (p = 0.039); the three sets that did not serve to find B (fresh, fresh-2, fresh-3) gained 9, lost 3
(p = 0.15; on fresh alone B gained 1 and lost 2); all five sets gained 18, lost 4 (p = 0.004, but the first two sets are where B was found). Against "always
EXPLAINED" on fresh-3: gained 2, lost 1.

*Wrong GENUINE calls.* Across the five sets the shipped Stage 7 calls a pair GENUINE 5 times in 250 (2 %), every time wrongly (first 50: 2, held-out: 1, fresh: 2;
none on fresh-2 and fresh-3); the Gemma 12B judge does it 42 times (17 %), 40 of them wrongly. The five sets hold 3 real contradictions (two in held-out, one in
fresh-3); the shipped Stage 7 finds none of them, the Gemma judge finds two of them (among its false calls).

*Which condition explains the difference.* Where the key says EXPLAINED and the system names conditions, the one it names equals the key's ("dev" and "test" counted as one) for 114 of 195 pairs (58 %; per set 54, 63, 62, 46, 66 %) against 56 of 157 (36 %; 40, 45, 38, 26, 30 %) for the Gemma 12B judge. That is 1.6 times the judge's rate and **below 85 %**: the stored profiles often lack the condition (a size or a setting recorded for one paper only) or state it differently from the key. With partial credit (mean Jaccard overlap between the named set and the key's set) the figures are 75 % (per set 71, 78, 78, 68, 80) against 54 % for the judge (53, 58, 55, 47, 58), also below 85 %.

*Limits.* The base rate of fresh-3 is 90 %, so accuracy cannot separate the systems there and the second line of evidence is macro-F1 and the false alarms; one
labeller, the same that wrote the rules; the pairs come from the same GLUE / SQuAD table families as before; the one genuine pair of fresh-3 (XLNet-large RTE on dev:
83.8 in the RoBERTa paper against 85.9 in the DeBERTa paper, two versions of the same result) is read by Stage 7 as a difference of setting ("single-task vs dev set");
the NOT COMPARABLE class of fresh-3 is small (4 pairs), so its macro-F1 moves by a few points with a single pair.

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

### 4.7 Retrieval and questions the papers cannot answer, against the gold evidence pages (9 Oct)

**Verdict: the page that holds the answer reaches the answer for 86 % of the gold questions and a question about something that is not in the corpus ends with a warning in 95 % of the cases. The reranker's own top 5 is the weak spot (75 %), which Stage 6 makes up for.**
The retrieval test of 1 October (`scripts/eval_retrieval.py`, 190 questions made from the profile store) is circular: a chunk counts as relevant when the same store says so. Here the relevant chunks are the ones on the pages that the key names as evidence for each question (written from the PDFs, field `evidence` of `eval/labels/questions_gold.jsonl`).

| Measure (28 gold questions whose evidence names a paper and a page; two are skipped: a bare paper list and an absence) | Result |
|---|---|
| Raw question as the only query, no LLM (a floor: the pipeline also rewrites and splits the question): the page is among the 20 fused candidates | 96.4 % |
| ... in the reranker's top 5 | 75.0 % (covered 90.0 %, one condition missing 87.5 %, partly covered 50.0 %); right paper in the top 5: 89.3 %; MRR 0.49 |
| ... kept by the reranker's threshold (up to 10 passages) | 96.4 % |
| **End to end** (the whole online path): the page is among the sources the answer is written from | **85.7 %** (24 of 28): covered 100 % (10), one missing 100 % (8), partly covered 60 % (10); the right paper 96.4 %; 5.2 sources on average |
| **Questions about things that are not in the corpus** (the 20 of `scripts/eval_retrieval.py`), end to end: the answer carries a warning | **95 %** (19 of 20); the scope warning alone 19 of 20, the reranker's weak-evidence flag alone 14 of 20 (70 %) |

The four end-to-end misses are comparison questions with a part the papers do not cover (GPT-3 or GPT-4 not evaluated on that data set, no Tamil column): the page of the covered side was not among the five sources, and for "BERT-base against GPT-3 on SST-2" the BERT paper is not among them at all. The one out-of-corpus question without a warning ("What BLEU does Mistral 7B get on WMT22 English-German?") got no conditions from Stage 1 ("conditions = none"), so Stage 6 had nothing to check: a Stage 1 failure, found by this test and not fixed.
Reading: retrieval is not the bottleneck (the page is among the 20 candidates 96 % of the time). A cross-encoder reads results tables poorly, so the reranker's order is the weak spot (75 % in its top 5), and the pipeline compensates through the passages that Stage 6 adds when they record a missing condition; the questions that compare two systems share five sources between two sides and suffer most (60 %).
Limits: 28 and 20 questions; the evidence pages were chosen by the same AI that wrote the questions and are one sufficient page, not the only one, so the numbers are a lower bound; the earlier store-made figures stay as they were (hit@20 95.3 %, top 5 90.0 %, language + task questions 60.0 % at the top 5), they just measure something circular. Scripts: `scripts/eval_retrieval_gold.py`, `scripts/eval_sources_gold.py`, `scripts/eval_absent_flagged.py` (the last two need the API running); results `eval/labels/retrieval_gold.json`, `sources_gold.json`, `absent_flagged.json`.

### 4.7b A stronger reranker and a Stage 1 fix (10 Oct; made after error analysis, so development numbers)

**Verdict: replacing the Stage 5 cross-encoder by `BAAI/bge-reranker-base` (the "stronger" fallback of `docs/backup_models_huggingface.md`) raises the reranker's own top-5 hit rate on the gold evidence pages from 75.0 % to 85.7 % and, on 189 store-made questions, from 87.8 % to 93.1 % (language + task questions from 60 % to 80 %). The end-to-end share of questions whose evidence page is among the answer's sources does not change (85.7 %; one question lost, one gained), so the gain is shown for ranking only. A Stage 1 fix lifts the questions about things that are not in the corpus and end with a warning from 19 to 20 of 20.**

Reranker comparison, same 20 fused candidates and the raw question as the only query (`scripts/eval_rerankers.py`, 28 gold questions; no LLM):

| Cross-encoder | top 3 | top 5 | top 10 | MRR | right paper in top 5 | top 5 by type (covered / one missing / partly covered) | s per question |
|---|---|---|---|---|---|---|---|
| ms-marco-MiniLM-L-6 (was) | 67.9 | **75.0** | 96.4 | 0.494 | 89.3 | 90 / 87.5 / 50 | 0.07 |
| ms-marco-MiniLM-L-12 | 60.7 | 71.4 | 92.9 | 0.489 | 82.1 | 80 / 87.5 / 50 | 0.09 |
| **bge-reranker-base** (now) | 75.0 | **85.7** | 96.4 | 0.580 | 92.9 | 90 / 100 / 70 | 0.22 |

Against the old one, bge-reranker-base gains the top 5 on 5 questions and loses it on 2 (p = 0.45: 28 questions cannot confirm it). The 189 store-made questions (`scripts/eval_retrieval.py`, circular because the same store defines "relevant", but larger) agree: top 5 87.8 % to 93.1 %, MRR 0.778 to 0.787, language + task questions (40) 60.0 % to 80.0 %, model + dataset questions (149) 95.3 % to 96.6 %. The model scores on another scale: bge-reranker-base returns raw logits only when told not to squash them (`activation_fn=Identity`, now set for every model), and its keep-threshold was swept again: **-3.0** keeps 7.2 passages per question on average, as the old model did at -2.0, with a hit rate of 92.1 % against 90.5 %. Stage 7's text-only path keeps ms-marco-MiniLM (`[models] text_relevance`), because its fixed relevance cut-off was calibrated on that model and Stage 7 is not to change.

End to end (the API, 28 gold questions, `scripts/eval_sources_gold.py`): the evidence page is among the sources for 24 of 28 (85.7 %) before and after; covered questions 100 % to 90 %, partly covered 60 % to 70 %, right paper 96.4 % to 92.9 %; median 10.1 s to 10.6 s. One covered question (B-AI-17) was lost and one comparison (B-AI-26) was gained: noise at this size. The out-of-corpus questions' own weak-evidence flag falls from 70 % to 50 % with the new threshold; the scope warning of Stage 6 is what catches them (below).

**Stage 1: a runaway list and hyphenated languages.** "What BLEU does Mistral 7B get on WMT22 English-German?" was the one out-of-corpus question without a warning (section 4.7). The cause is in two places: the language model wrote an endless `other_languages` list ("German-English", "German-English-German-English-...") until the token cap cut the JSON, every time (temperature 0), so Stage 1 fell back to the keyword heuristic; and the fallback that reads languages from the question could not see "English-German" (a hyphen next to a name blocked the match). Three general changes, each with a unit test (230 tests pass): (1) when the full reading is invalid, one second reading asks for the single-valued conditions only; (2) a language next to a hyphen or a slash is found; (3) further models and languages written in the question that the model dropped are added from the question ("mT5 and XLM-R on XQuAD for Arabic and Thai" had come back with Thai only). Effect on the 30 gold questions (Stage 1 conditions): language recall 0.94 to 1.00, all conditions F1 0.923 to 0.929, scope-warning F1 unchanged (97.4 %); questions about things that are not in the corpus and end with a warning **19 to 20 of 20 (100 %)**, the scope warning alone 20 of 20 (`eval/labels/absent_flagged.oct10.json`). The answer test is unchanged: 8 of 9 with and without the profile stages, 0 of 9 for the language model alone.

Limits: both changes were found and tuned on these question sets, so the after-numbers are development numbers; 28 and 20 questions; the reranker's end-to-end effect is not shown; the Stage 1 fix was found on one question, and the second reading has not been tried on other kinds of invalid output. Scripts: `scripts/eval_rerankers.py`, `scripts/eval_retrieval.py` (its threshold grid now runs from -9 to 7), `scripts/eval_retrieval_gold.py`, `scripts/eval_sources_gold.py`, `scripts/eval_absent_flagged.py`; results `eval/labels/rerankers_gold.json`, `retrieval_gold.oct10.json`, `sources_gold.oct10.json`, `absent_flagged.oct10.json`, `questions_gold.oct10_main.scores.json`, `answer_quality_gold_b.oct10.json`.

### 4.7c The whole system on the repaired store, and what it exposed (11 Oct)

**Verdict: the repaired store does not by itself improve the system-level numbers; on the first run it lowered some of them. The causes were two general
weaknesses that the repaired names and cards exposed (a named setting was dropped or not required together with model and data set; cards did not name the data
set version) and a shift in the ranking of the retrieval cards. After three fixes (made after reading single questions of the 30 gold questions, so
development numbers) Stage 6 and the answers are at the 10 Oct level: scope warning 19 / 1 / 0, F1 97.4, condition named 19 of 19 (10 Oct 19 / 1 / 0, F1 97.4, condition named 17 of 19), answers 8 / 7 / 0 of 9 on the corrected key
(10 Oct 8 / 8 / 0 of 9; run-to-run noise of one or two questions), silver regression 32 of 32, the gold page is among the answer's sources for
85.7 % of the 28 questions (85.7 % before) and 100 % of the out-of-corpus questions end with a warning. The retrieval
stage alone is lower: the gold page is in the top 5 for 78.6 % against 85.7 %.**

| Measure (30 gold questions; 28 for retrieval and sources; AI-annotated key) | 10 Oct, raw store | 11 Oct, repaired store, first run | 11 Oct, after the three fixes |
|---|---|---|---|
| Stage 1 conditions, micro-F1 (setting recall) | 92.9 (0.40) | 92.9 (0.40) | 95.0 (1.00) |
| Stage 6 scope warning: TP / FP / FN | 19 / 1 / 0, F1 97.4, condition named 17 of 19 | 18 / 1 / 1, F1 94.7, condition named 17 of 19 | 19 / 1 / 0, F1 97.4, condition named 19 of 19 |
| Silver regression (32 store-made scope questions) | 32 of 32 | 32 of 32 | 32 of 32 |
| Retrieval: gold page in the top 20 / top 5 / right paper in the top 5 (%) | 96.4 / 85.7 / 92.9 | 92.9 / 78.6 / 89.3 | 92.9 / 78.6 / 89.3 |
| Sources: gold page / right paper among the answer's sources (%) | 85.7 / 92.9 | not run | 85.7 / 92.9 |
| Out-of-corpus questions that end with a warning (20) | 100.0 % | not run | 100.0 % |
| Answers right, corrected key: full / without stages C, 6, 7 / LLM alone (9 questions) | 8 / 8 / 0 of 9 | 7 / 8 / 0 of 9 | 8 / 7 / 0 of 9 |

*What the first run showed.* (1) **A data set written correctly is a data set that can be missed.** The repair turned "ellaSwag" into "HellaSwag"; the
question "What accuracy does LLaMA 65B get on HellaSwag in the 5-shot setting?" had been warned before only because the data set name did not match
(a warning for the wrong reason: it named the data set as missing). With the name right, Stage 1 turned out to drop "5-shot" (`setting_tags` knows no N-shot
tag, so `clean_conditions` removed it as "not an evaluation setting"), and Stage 6 would have accepted the setting anyway because the paper says "5-shot" in
a passage on MMLU. Fixes: an N-shot setting named in the question is a condition (`_N_SHOT`), a task that is only a setting word is filed as the setting, and
the joint check of Stage 6 now includes a named setting when it is a regime (zero-shot, N-shot, few-shot, fine-tuned, translate-*; `[features] joint_setting`, on),
not a split (dev / test): "LLaMA 65B on HellaSwag in the 5-shot setting" is reported as not recorded together, naming the recorded zero-shot. The setting recall of Stage 1 on these questions rose from 0.40 to 1.00.
(2) **Cards that do not name the version cannot tell SQuAD 1.1 from SQuAD 2.0.** With the repaired settings in the cards ("single test set" ...) the cross-encoder
scored the two tables of the BERT paper 2.48 (SQuAD 2.0) and 2.46 (SQuAD 1.1) for "How well does BERT perform on the SQuAD 2.0 dataset?" (2.99 and 1.84 before),
and the answer's sources held the 1.1 table but not the 2.0 table; the answer quoted SQuAD 1.1 numbers (that question went from right to wrong, the full
pipeline from 8 to 7 of 9). Cards now name the data set with its version ("SQuAD 2.0").
(3) **The repaired cards change which pages rank in the top 5.** Against the 28 gold pages the top-5 hit falls from 85.7 to 78.6 %
(top 20: 96.4 to 92.9 %): three questions lose the page (one of them out of the top 20, two by one rank), one gains it. Cards built from the
raw fields with today's code give exactly the 10 Oct figures (85.7 %), cards that list both the raw and the repaired values 82.1 %. The question that left
the top 20 (B-AI-18) asks for a zero-shot result of BERT-base on MNLI: after the repair many XNLI chunks really record zero-shot, so the question now finds those
chunks and not the fine-tuned BERT table that the key counts as the evidence; B-AI-23 and B-AI-29 flip from rank 5 to 6. The end-to-end figure is the share of
questions whose gold page is among the sources the answer is written from: 85.7 % (85.7 % before).

*What the answer numbers can and cannot say.* The nine answerable questions are scored by the numbers the key lists; one question is one eleventh of the
percentage and the answer model is sampled at temperature 0.2: the "without stages C, 6, 7" system alone scored 5, 5, 5, 4, 5 of 9 on the uncorrected key in
5 runs of 10-11 Oct (8, 8, 7, 6, 7 on the corrected key) with no change of code that concerns it, so differences of one or two questions between runs are noise. The
claim that stands is the earlier one: the profile stages do not reduce answer correctness, and the language model alone gets none of the nine.

*Ten questions outside the gold set (11 Oct; a qualitative probe of the setting rule, not a test).* Seven name a result that the papers have, two name one
that they do not have, one is undecided. Six of the seven were not warned (GPT-3 few-shot on TriviaQA; BERT-large F1 on SQuAD 1.1 on the dev set; RoBERTa fine-tuned
on MNLI; MuRIL on IndicXNLI for Kannada on the test set; LLaMA 65B 5-shot on NaturalQuestions; XLM-R on XNLI for Hindi zero-shot, which is covered only through the
"Cross-lingual Transfer" alias: 2 profiles with it, 0 without). One was warned although the result exists: mBERT on XNLI for Hindi in the zero-shot setting, because the
only mBERT rows of that data set and language that the store recognises are labelled "test set" (MuRIL paper) or have no setting (mT5 paper), and the XLM-R paper's
mBERT rows are labelled "Devlin et al. (2018)" and are not recognised as mBERT (a known limitation). The two questions about results the papers do not have
(LLaMA 65B on HellaSwag 5-shot, BERT-base on MNLI zero-shot) were warned with the recorded settings named; the undecided one (XLM-R on MLQA for Hindi, translate-train)
was warned, probably rightly. The first version of the rule required a dev / test split to be recorded jointly as well and warned falsely on the MuRIL question
(no setting stored): the rule was restricted to regime settings (zero-shot, N-shot, few-shot, fine-tuned, translate-*). So 8 of 10 probes came out as the papers say, one
probably did, and the one miss is a store limitation, not a matching error.

*Limits.* The three fixes were found by reading single questions of the same 30 (28) questions that the earlier fixes were developed on; the numbers of the last
column are therefore development numbers and not a held-out test. The key of job B is v2 (section 3, item 2). Only questions that name a setting
(five of the 30) are affected by the joint-setting check; its effect on the many other kinds of question is not measured by this set. The stored `setting`
field is noisy (free text; 98.5 % filled in the raw store but often "fine-tuned", "test set", "single model"), so a question that names a setting is the case in which
a false scope warning is most likely; `[features] joint_setting = false` restores the 10 Oct joint check.

*Stage 6 against the baselines again (same 30 questions, key v2, final code).*

| System | TP / FP / FN | precision | recall | F1 | false warning rate % | right condition named |
|---|---|---|---|---|---|---|
| plain RAG | 0 / 0 / 19 | 0.0 | 0.0 | 0.0 | 0 | None |
| abstain on weak retrieval | 0 / 0 / 19 | 0.0 | 0.0 | 0.0 | 0 | None |
| LLM judge (Gemma 12B, Sufficient-Context style) | 16 / 1 / 3 | 94.1 | 84.2 | 88.9 | 9 | None |
| Stage 6 of 30 Sep (no joint check) | 4 / 0 / 15 | 100.0 | 21.1 | 34.8 | 0 | 3 |
| Stage 6 + joint check | 19 / 1 / 0 | 95.0 | 100.0 | 97.4 | 9 | 19 |
| Stage 6 shipped (every switch on) | 19 / 1 / 0 | 95.0 | 100.0 | 97.4 | 9 | 19 |

Paired sign tests over the 30 questions, the shipped Stage 6 against the LLM judge 4 won / 1 lost (p = 0.375); the 30 September version 15 won / 1 lost (p < 0.001); plain RAG 19 won / 1 lost (p < 0.001). The LLM judge and Stage 6 are still within the noise of 30 questions, the earlier version without the joint check is clearly worse, and what the shipped Stage 6 adds over the judge is the named missing condition (19 of 19; the judge only says yes or no).

### 4.8 Faithfulness (RAGAS-style) and the optional rewrite loop (9-10 Oct)

**Verdict: the numbers in the shipped answers are grounded. In 121 decimal numbers only one is not found in the text the answer was written from (a formatting variant, "92.70" for a recorded 92.7; a check with no language model), the Gemma 12B judge finds 90 % of the statements supported and the NLI critic of Stage 9 accepts 85 % of the claims. The independent judge (Llama 3.1 8B) gave 0.60, but a hand check showed that it misreads results tables, so that figure is not evidence about the answers. The optional rewrite loop has no demonstrated benefit: it raises the score of the judge it uses, no judge-free measure moves, and it doubles the time per question. It stays an optional switch, off by default.**

What was built (`src/cgrag/pipeline/faithfulness.py`; the `ragas` package is **not** installed, so these are re-implementations of the two RAGAS definitions with Ollama models, not comparable with published RAGAS numbers):
- *Faithfulness*: the judge splits the answer into statements and says for each whether the passages support it; score = supported / checked. A statement that only says what the papers lack is not checked; an answer with nothing to check gets no score. The judge sees the text of the passages the answer was written from plus the recorded results the writer was shown (the longest context was 14,130 of 16,000 allowed characters, so nothing was cut).
- *Answer relevancy*: the judge writes three questions the answer could belong to; score = their mean cosine (BGE-M3) with the real question, and 0 for an answer the judge reads as non-committal.
- *The loop* (`[features] ragas_loop`, off; `[ragas]`: threshold 0.80, at most 3 rewrites; live step `9b_ragas`): after the critic the answer is scored; below the threshold the writer gets the unsupported statements as feedback and writes again; the best text is kept. This is the architecture's original runtime loop, which the design had replaced by the NLI critic with one retry.

Setup: the 30 gold questions, three answer sets from the same code, one run each: **loop off** (the shipped configuration), **loop on with Gemma 12B as its own judge**, **loop on with Llama 3.1 8B as the judge** (resident next to the 12B and the 2B agent). Every set was then scored offline by both judges (`scripts/eval_ragas.py`; 61 minutes on this PC with Ollama in a 2-slot profile: 12B 9.0 GB, 2B 1.9 GB, Llama 7.0 GB, the pipeline's own models plus the desktop about 4.8 GB, 22.6 of 24.5 GB in all).

| Judge (mean faithfulness; answers at 0.80 or more) | Loop off | Loop on, Gemma judges | Loop on, Llama judges |
|---|---|---|---|
| **Gemma 4 12B** (the writer's own family) | **0.904** (21 of 27) | 0.956 (27 of 28); +8 / -4 against loop off, p = 0.39; *circular: the loop used this judge* | 0.904 (22 of 28); +7 / -6, p = 1.0, difference +0.01 (95 % CI -0.07 to +0.08); *cross-check* |
| **Llama 3.1 8B** (another family; **found unreliable, see 2**) | 0.600 (11 of 28) | 0.663 (12 of 29); +8 / -7, p = 1.0; *cross-check* | 0.818 (22 of 30); +14 / -4, p = 0.031; *circular: the loop used this judge* |

(The scored count is below 30 where an answer only says what the papers lack. Paired sign test over the questions both sets have scores for.)

| | Loop off | Loop, Gemma judges | Loop, Llama judges |
|---|---|---|---|
| **Decimal numbers in the answer that occur in its sources (no language model)** | **99.2 %** (120 of 121; 24 of 25 answers fully grounded) | 100 % (127 of 127; 24 of 24) | 99.1 % (106 of 107; 24 of 25) |
| Right numbers in the 9 answerable gold questions (corrected key, no language model) | 8 of 9 | 9 of 9 | 9 of 9 |
| Seconds per question, median (90th percentile) | 10.1 (14.1) | 21.1 (39.5) | 21.8 (43.8) |
| Answers rewritten (rewrites in all) | 0 | 3 of 30 (6) | 15 of 30 (34) |
| Answer relevancy, Gemma judge / Llama judge | 0.49 / 0.69 | 0.48 / 0.73 | 0.56 / 0.72 |

Reading:
1. **Which figures can be trusted.** The number check has no judge: 99 % of the decimal numbers in the shipped answers occur in the passages and recorded results the answer was written from (the one miss is "92.70" for a recorded 92.7). It cannot tell a right number from one that belongs to another model; for that there are the Gemma judge (0.90), the NLI critic (85 % of the claims, section 4.6) and the number-matching test against the key (8 of 9, section 4.6). The Gemma judge is of the writer's own family and may favour its own text; on the three answers whose statements, verdicts and stated reasons I read in detail (B-AI-03, B-AI-06, B-AI-11) its verdicts matched the tables cell by cell. Three figures from three different kinds of check are above 0.80; none of them is independent of the system in the way a human reader would be.
2. **The independent judge failed a validity check, so its numbers are not used as evidence.** A hand check (10 Oct) of the statements Llama 3.1 8B called "unsupported" found that in the answers read in detail they are in the context: for B-AI-11 it flagged "XLM-R obtains 71.5 accuracy on the IndicXNLI dataset for the Kannada language" although the recorded results contain "XLMR obtains 71.5 accuracy on IndicXNLI (single model, Kannada)"; for B-AI-03 it flagged the 35.0 that the table gives in the 5-shot column; for B-AI-06 it flagged every per-setting mT5-Large and XLM-R value (65.7 / 45.3, 66.1 / 48.1, 74.0 / 56.7, 66.5 / 47.7, 87.4 / 79.6), all of which stand verbatim in the passages. Its stated reasons show why: it does not match "XLMR" with "XLM-R", it reads the heading of the recorded results as a restriction ("not present in the recorded results"), it cannot assign a table column to a setting, it garbles the statements it extracts ("the 5-shot setting is the model setting"), and in one answer it returned 13 verdicts for 8 statements, which lowers the score. Gemma 12B read the same contexts correctly. **My first reading of these runs ("the independent judge is stricter than the writer's family") was wrong: it is less accurate.** The Llama column in the table is therefore kept as measured and labelled, not interpreted.
3. **The rewrite loop has no demonstrated benefit.** With Gemma as its own judge it rewrote 3 of 30 answers (the judge finds little to fix). With Llama as the judge it rewrote 15 of 30, but it was optimising the verdicts of a judge that misreads tables, and the cross-check by Gemma shows no change (0.904 to 0.904, CI -0.07 to +0.08). The number check moves by less than one number (99.2 % to 99.1 % to 100 %), the right numbers stay right (9 of 9 with either loop, 8 of 9 without; one question, p = 1.0), and the time per question doubles (median 10 to 21-22 s, 90th percentile up to 44 s). The shipped configuration already reaches 0.80 under the judge that reads tables correctly, so the loop is not needed.
4. **Answer relevancy is not a quality number here.** By definition an answer the judge reads as non-committal scores 0, and "the papers do not report this" is all or part of the right answer to the 19 questions that need a scope warning: Llama flags 3-4 of the 30 answers that way, Gemma 9-12.

Decision: the loop stays an optional, off-by-default switch; the shipped design (NLI critic, one retry) is unchanged.

Limits: one run per set (the model output is not exactly repeatable, section 5); 30 questions; **no valid independent judge was found**: the only independent one was a small 8B model that failed the hand check, and a larger judge of another family (a 14B model or more) was not tried; the hand check covers three answers in detail and three more at a glance, not a sample; the number check cannot see a right number attached to the wrong model; the metrics are our re-implementation, not the `ragas` package, and are not comparable with other papers'. The first two questions were also run once as a smoke test of the scripts; that test is not part of the numbers.
Scripts: `scripts/eval_ragas.py` (collect / judge / report), `scripts/analyze_ragas.py` (paired tests, cost, number checks); results `eval/labels/ragas_report.json`, `eval/labels/ragas_analysis.json` (aggregates only; the answers and per-question scores are private files in `data/labelling/private/ragas/`).

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

**10 Oct (after reading failures; development numbers, section 4.7b and 4.5d).** (1) Stage 1: a second reading with the single-valued conditions only when the full reading is invalid JSON; (2) a language next to a hyphen or slash is found ("English-German"); (3) further models / languages written in the question are added when the model dropped them. (4) Stage 5: `BAAI/bge-reranker-base` with raw logits and a threshold of -3.0 (Stage 7's text path keeps ms-marco-MiniLM). (5) Stage 7 policy B (`one_sided_conditions_explain`) built, tested on a fresh set and left OFF. (6) Table parser: a block heading cut at column borders (its last piece looks like a number) is a heading, not a result row. (7) **Stage 7 table grounding** (`src/cgrag/pipeline/grounding.py`, `[contradiction] table_grounding`, ON): the cards' own table cells are read again for split, system kind, size and for cards that do not fit them (section 4.5e). 240 unit tests pass.

**11 Oct (development numbers, sections 4.1b, 4.1c, 4.5f, 4.7c).** (0) Three fixes of Stage 1 / Stage 6 / the cards after the first system run on the repaired store (item 25): N-shot settings are conditions, a named setting must be recorded together with model and data set (`joint_setting`), cards name the data set version. (1) The profile repair and its overlay table (`[features] profile_repair`, on); (2) the table reader switches the column names at a header row that is repeated in the middle of a table and joins group labels printed vertically; (3) Stage 7 policy B (`one_sided_conditions_explain`) is ON together with table grounding; (4) `CGRAG_OVERLAY` merges one more configuration file (used for the benchmark's paths). 258 unit tests pass.

## 7. What these numbers can and cannot show

- **Independence.** One AI labeller who also helped build the system. The labels are checked against the PDFs, but a second reader could disagree on borderline cases (Stage 7's "not comparable" class and the lenient / strict question in job A are the soft spots). The second job-A sheet and the filled teammate sheets (except the second-labeller pair sheets, which are Gemini's) were made by the same labeller with the same rules; both job-C third looks are not blind and sided with the first pass on most pairs (8 of 10 and 6 of 7).
  The agreement between the two AI labellers on job C is 80 % (kappa 0.40) and 86 % (kappa 0.52), both below the 0.6 target. Gemini had the PDFs of each batch in both rounds (the project lead's statement, not logged at the time), yet several of its disagreements trace to the sheet excerpts, which sometimes omit a table's block heading (section 3, item 15); it seems to have worked from the excerpts more than from the pages.
- **Small samples.** 30 questions, 9 answerable questions with numbers, 50 pairs, 320 profiles. Confidence intervals (Wilson, 95 %) are given and are wide: one question moves the Stage 6 F1 by about three points. A difference of one or two questions is not a difference.
- **Stage 7 has almost no genuine contradiction in its samples.** The 28 papers are on different tasks, and the pairs came from the numeric trigger; the first key has 0 genuine pairs and the held-out key 2 (OpenAI GPT's GLUE test scores, one labeller unsure). So the tests measure false alarms and the condition named, **not** the ability to find a real contradiction (0 of 2 found; the third fresh set of 11 Oct adds one more, an XLNet figure that exists in two versions, also read as a setting difference: 0 of 3 over five sets). Accuracy alone is a weak yardstick: always answering EXPLAINED would score 82 % on both sets.
- **The held-out result is one sample of 50 pairs.** 58 % has a 95 % CI of 44-71 %, and the Gemma judge's 66 % lies inside it; the false-alarm advantage (7 % against 21 %) is the only difference that is statistically clear (p = 0.003, 100 pairs). The held-out key itself was settled by a third look that is not blind to the labels.
- **"No effect" in the ablation means "not detected on 30 questions".** The agents, escalation, cards and profile-guided retrieval were built for harder or larger question sets and for latency; this set cannot see them.
- **The answer-quality test is small and strict.** The key lists the numbers of one paper; an answer that cites another paper's number counts as a miss. Nine questions; the corrected key (8 of 9) was fixed after the first scoring (section 3, item 16), so it is not held-out.
- **Not done:** an extraction baseline on the 320 profiles of key A (the baseline exists only on the public benchmark, 4.1c); the `ragas` package itself (the RAGAS-style metrics were re-implemented, section 4.8); a larger independent judge (a 14B model or more of another family; the hand check of the 8B judge failed, section 4.8); a human second labeller; a user study of the interface (it was checked with headless Chrome scripts and looked at by the project lead only).
- **Stage 7's accuracy is judged against a base rate of 80 %.** 201 of the 250 labelled pairs are EXPLAINED, so a rule that always says EXPLAINED scores 80.4 %. The shipped Stage 7 (table grounding + policy B on the repaired store) reaches 88.0 % over the five sets (82.5 % over the first four before the repair and policy B; the clean 10 Oct set gave 88 %, the clean 11 Oct set 92 % against 90 % for the trivial rule); its evidence of skill is macro-F1, the not-comparable pairs found (25 of 46) and the false alarms avoided, not accuracy. What is left (4.5e, 4.5f): metric mixes (F1 against accuracy), numbers read from prose, a figure that exists in two versions, and the stored profiles themselves (repaired by an overlay, not re-extracted).
- **The fresh pair sets have one labeller.** No second labeller and no kappa; the same AI helped build the system; 9 labels of medium confidence in each. Their pairs come from a handful of GLUE / SQuAD-style tables repeated across papers, and the last set reuses profiles of the earlier ones in new pairs (4.5e): grounding is shown to generalise to new pairs of this corpus, not to new table layouts.
- **The system-level numbers of 11 Oct are development numbers.** The three fixes of section 4.7c were found on the same 30 gold questions; the setting-bearing questions are five, so the joint-setting check has been exercised on five cases only.
- **The repair is judged by the AI that wrote it.** Field by field, with the stored and the repaired version in random order, but the rules are known to the labeller; the human gold of MetaLead (section 4.1c) is the independent check of names, and there the repaired numbers are development numbers. The share of completely right profiles (60.6 % lenient, 39.4 % strict on the clean sample) is below the 85 % that was asked for; the remaining errors are named in 4.1b.
- **A group, subject or subset of a data set is not recorded** (MMLU subjects, BOLD groups, RACE middle / high, GLUE diagnostic categories); two results of one table that differ only in it look the same to Stage 6 and Stage 7 (a `subset` field is not built).
- **The plain-LLM baseline is one configuration** (windows of 6,000 characters, one prompt, one local model); a model with a long context window or an agentic reader could do better, and the benchmark authors' reference models were not re-run.
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
# 10 Oct: rerankers, Stage 1 and the fresh Stage 7 set
python scripts/eval_rerankers.py                                       # 3 cross-encoders on the 28 gold questions (API stopped; the first run downloads bge-reranker-base, 1.1 GB)
CGRAG_CONFIG=<a copy of config.toml that names another reranker> python scripts/eval_retrieval.py --cases <earlier cases file> --out <file>     # silver sweep, same questions
python scripts/make_heldout_pairs.py --pairs 50 --seed 1017 --prefix F --name fresh --also-exclude data/labelling/private/job_C_heldout_key.json --people first
python scripts/fill_job_c_from_ai.py --answers <jsonl> --sheet data/labelling/C_fresh_pairs_first.xlsx --model "..." --date ... --tag fresh-first --role "FRESH FIRST PASS"
python scripts/run_pair_baselines.py --key data/labelling/private/job_C_fresh_key.json --out data/labelling/private/job_C_fresh_baselines.json     # needs Ollama; ~105 s
python scripts/make_heldout_pairs.py --pairs 50 --seed 1018 --prefix G --name fresh2 --allow-profile-reuse --reuse-subjects --also-exclude data/labelling/private/job_C_heldout_key.json data/labelling/private/job_C_fresh_key.json --people first     # the final clean set
python scripts/fill_job_c_from_ai.py --answers data/labelling/private/fresh2_first_pass.jsonl --sheet data/labelling/C_fresh2_pairs_first.xlsx --model "..." --date ... --tag fresh2-first --role "FRESH-2 FIRST PASS"
python scripts/run_pair_baselines.py --key data/labelling/private/job_C_fresh2_key.json --out data/labelling/private/job_C_fresh2_baselines.json
python scripts/score_stage7_variants.py                                # replays classify on all 200 pairs, with and without fix A / policy B / table grounding, paired sign tests -> eval/labels/stage7_variants.json
# retrieval against the gold evidence pages (9 Oct); the first needs no API (read a COPY of data/index/chroma while the API runs), the last two need it
python scripts/eval_retrieval_gold.py [--chroma <copy of data/index/chroma>]
python scripts/eval_sources_gold.py                                    # 28 questions, about 5 minutes
python scripts/eval_absent_flagged.py                                  # 20 questions, about 4 minutes
# faithfulness and the rewrite loop (9-10 Oct): stop the API first; Ollama with OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=3; `ollama pull llama3.1:8b`; about 1 hour in all
python scripts/eval_ragas.py collect --tag loop_off                                      # 30 answers, loop off (shipped configuration)
python scripts/eval_ragas.py collect --tag loop_same --loop                              # loop on, the answer model judges itself
python scripts/eval_ragas.py collect --tag loop_llama --loop --judge-model llama3.1:8b   # loop on, the independent judge resident (about 22.6 of 24.5 GB)
python scripts/eval_ragas.py judge --tag <tag> --judge llama3.1:8b                       # and again with --judge gemma4:12b, for each of the three tags
python scripts/eval_ragas.py report --tags loop_off loop_same loop_llama --judge llama3.1:8b   # -> eval/labels/ragas_report.json (and the same for gemma4:12b)
python scripts/analyze_ragas.py                                                          # paired tests, cost, number check -> eval/labels/ragas_analysis.json
# system runs (Ollama up, no other pipeline process, ~45 minutes in all)
python scripts/eval_questions.py eval/labels/questions_gold.jsonl --tag final_main              # and --set KEY=VALUE for each ablation
python scripts/eval_scope_baselines.py --gold eval/labels/questions_gold.jsonl --out eval/labels/scope_baselines_gold.final.json
python scripts/eval_answers.py --gold-b eval/labels/questions_gold.jsonl --out eval/labels/answer_quality_gold_b.final.json
python scripts/ablate_stage6.py --configs "+escalation"                                          # silver regression, must stay 32 / 32
# 11 Oct: the profile repair, its two clean test sets, the public benchmark, the plain-LLM baseline and the third fresh pair set (docs/reproducibility.md has the order)
python scripts/repair_store.py --stats                                  # what the repair changes; --apply writes the overlay (raw rows untouched), --clear removes it; then python scripts/reindex_cards.py
python scripts/make_extraction_testset.py --seed 1110                   # E1: 120 profiles of the 28 papers (with CGRAG_OVERLAY=config/bench_metalead.toml and --seed 2011 --out data/labelling/private/extraction_test2: E2, 100 profiles of the benchmark papers)
python scripts/extraction_test_ab.py sheet --seed 7 [--dir <set>]        # blind A / B sheet of stored vs repaired fields; label it; then: python scripts/extraction_test_ab.py score [--dir <set>] -> eval/labels/extraction_repair_<set>.json
python scripts/download_metalead.py                                     # 43 papers + 3,568 annotated results -> data/bench/metalead (git-ignored); CGRAG_OVERLAY=config/bench_metalead.toml python scripts/ingest.py reads them (about 2 h)
CGRAG_OVERLAY=config/bench_metalead.toml python scripts/eval_metalead.py   # coverage / agreement / precision, stored and repaired -> eval/labels/metalead_eval.json
python scripts/baseline_llm_extract.py --workers 8                      # plain-LLM baseline (needs Ollama, about 1 h)
python scripts/eval_metalead.py --pred data/bench/metalead/baseline_gemma.json --name baseline_gemma12b_windows_split --out eval/labels/metalead_eval_baseline.json
python scripts/make_heldout_pairs.py --pairs 50 --seed 1019 --prefix J --name fresh3 --allow-profile-reuse --reuse-subjects --also-exclude <the four earlier keys> --people first   # fresh-3, from the repaired store
PROFILE_REPAIR=on python scripts/score_stage7_variants.py               # all 250 pairs through the repair overlay (default = raw store) -> eval/labels/stage7_variants.json
python scripts/reindex_cards.py --cards raw|repaired|union             # which profile fields the retrieval cards are built from (the A / B of section 4.7c); an index copy: --chroma / --bm25 / --profiles
python scripts/eval_retrieval_gold.py --chroma <index>/chroma --bm25 <index>/bm25.pkl --out eval/labels/retrieval_gold.<tag>.json --tag <tag>     # retrieval of the 28 gold pages on any index copy
python scripts/rescore_answers_gold_b.py <answers.json> <out.json>     # the corrected key (section 3, item 16) for any run of eval_answers.py
CGRAG_OVERLAY=config/raw_cards_oct11.toml python scripts/eval_questions.py ...     # retrieve with the pre-repair cards, read the repaired profiles (ablation overlay)
```

If teammates (or anyone else) ever deliver real label files, put the workbooks into `data/labelling/done/` and run the same scorers
(`score_labels.py status|A|B|C`); for job C two files give Cohen's kappa. The frozen files are `eval/labels/*.frozen_ca007a6.json` and the `v2_main` / `no_*` result files;
the clean re-run is `*.final*`. Raw results, keys and freeze records are in `eval/labels/`.
