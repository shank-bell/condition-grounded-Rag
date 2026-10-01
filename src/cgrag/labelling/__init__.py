"""Labelling kit: the Excel sheets the team fills in (jobs A, B, C) and the scorer that turns them into the evaluation's answer key.

  A  profile check     is what the extractor wrote correct?             -> field-level precision / recall / F1
  B  questions         what should the system do with this question?    -> intent / complexity / scope-warning accuracy
  C  result pairs      do two papers really disagree, or is it explained? -> Cohen's kappa between people, Stage 7 accuracy

sampling.py picks the rows, xl.py is the shared look of the workbooks, sheets.py builds them, score.py reads them back.
Labels are an answer key for scoring the finished system; nothing is trained on them.
"""
