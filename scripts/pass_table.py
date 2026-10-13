"""Print the comparison table of whole evaluation passes (docs/evaluation_ai_annotated.md, section 4.7f) from the result files of eval/labels/.

  python scripts/pass_table.py oct11g oct11i oct11j oct11k oct12a          # Markdown table, one column per pass tag
  python scripts/pass_table.py oct11g oct12a --json                          # the same numbers as JSON

A pass = scripts/eval_retrieval_gold.py, eval_questions.py (`<tag>_main`), ablate_stage6.py, eval_answers.py + rescore_answers_gold_b.py, eval_absent_flagged.py,
eval_sources_gold.py, eval_scope_baselines.py, each written with the tag in its file name (`retrieval_gold.<tag>.json`, `questions_gold.<tag>_main.scores.json`,
`ablate_stage6.<tag>.json`, `answer_quality_gold_b.<tag>[.corrected_key].json`, `absent_flagged.<tag>.json`, `sources_gold.<tag>.json`,
`scope_baselines_gold.<tag>.json`). The Stage 7 replay of 300 labelled pairs is not tag-specific (it reads the store as it is): give it with --stage7 "267 of 300".
A file that does not exist for a tag gives "n/a".
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

L = Path(__file__).resolve().parents[1] / "eval" / "labels"


def load(name: str):
    p = L / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def sign_p(a: int, b: int) -> float:
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def column(tag: str) -> dict:
    out: dict = {}
    q = load(f"questions_gold.{tag}_main.scores.json")
    if q:
        sw, c = q["scope_warning"], q["conditions"]["ALL"]
        out["scope"] = f"{sw['tp']} / {sw['fp']} / {sw['fn']}, F1 {100 * sw['f1']:.1f}, named {sw['right_condition_named']} of {sw['expected_warnings']}"
        out["stage1"] = f"{100 * c['f1']:.1f} (P {100 * c['precision']:.1f}, R {100 * c['recall']:.1f})"
        out["intent"] = f"{100 * q['intent']['accuracy']:.1f} / {100 * q['complexity']['accuracy']:.0f}"
        out["seconds"] = f"{q['seconds_per_question']['median']}"
    sb = load(f"scope_baselines_gold.{tag}.json")
    if sb:
        j = sb["scores"]["sufficient_context"]
        won = sum(bool(r["systems"]["stage6_full"]["warned"]) == bool(r["expect_warning"]) and bool(r["systems"]["sufficient_context"]["warned"]) != bool(r["expect_warning"])
                  for r in sb["rows"])
        lost = sum(bool(r["systems"]["sufficient_context"]["warned"]) == bool(r["expect_warning"]) and bool(r["systems"]["stage6_full"]["warned"]) != bool(r["expect_warning"])
                   for r in sb["rows"])
        out["judge"] = f"{j['tp']} / {j['fp']} / {j['fn']}, F1 {100 * j['f1']:.1f}; Stage 6 won {won}, lost {lost}, p = {sign_p(won, lost):.3f}"
    ab = load(f"ablate_stage6.{tag}.json")
    if ab:
        rows = ab["results"]["+escalation"]
        out["silver"] = f"{sum(bool(x['warned']) == bool(x['expect_warning']) for x in rows)} of {len(rows)}"
    r = load(f"retrieval_gold.{tag}.json")
    if r:
        s = r["summary"]
        out["retrieval"] = f"{s['hit@20']} / {s['hit@5(rerank)']} / {s['paper_hit@5']}"
    s = load(f"sources_gold.{tag}.json")
    if s:
        out["sources"] = f"{s['summary']['evidence_page_in_sources_%']} / {s['summary']['evidence_paper_in_sources_%']}"
    a = load(f"absent_flagged.{tag}.json")
    if a:
        out["absent"] = f"{round(a['summary']['questions'] * a['summary']['flagged_by_either_%'] / 100)} of {a['summary']['questions']}"
    ans, cor = load(f"answer_quality_gold_b.{tag}.json"), load(f"answer_quality_gold_b.{tag}.corrected_key.json")
    if ans:
        sc = ans["scores"]
        out["answers_raw"] = f"{sc['full']['correct']} / {sc['no_stage_c67']['correct']} / {sc['llm_only']['correct']}"
    if cor:
        c = cor["corrected"]
        out["answers_cor"] = f"{c['full']['correct']} / {c['no_stage_c67']['correct']} / {c['llm_only']['correct']}"
    return out


ROWS = [("Stage 6 scope warning TP / FP / FN", "scope"), ("LLM judge on the same questions (TP / FP / FN)", "judge"), ("Silver regression (32 store-made scope questions)", "silver"),
        ("Stage 1 conditions micro-F1", "stage1"), ("Stage 1 intent / complexity accuracy (%)", "intent"),
        ("Retrieval alone: gold page in top 20 / top 5 / right paper in top 5 (%)", "retrieval"), ("Sources: gold page / right paper among the answer's sources (%)", "sources"),
        ("Out-of-corpus questions that end with a warning", "absent"), ("Answers right of 9 (original key): full / without C, 6, 7 / LLM alone", "answers_raw"),
        ("Answers right of 9 (corrected key; not held-out)", "answers_cor"), ("Median seconds per question, Stages 1-6", "seconds")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cols = {t: column(t) for t in args.tags}
    if args.json:
        print(json.dumps(cols, indent=1, ensure_ascii=False))
        return
    print("| Measure (AI-annotated key) | " + " | ".join(f"`{t}`" for t in args.tags) + " |")
    print("|---|" + "---|" * len(args.tags))
    for label, k in ROWS:
        print(f"| {label} | " + " | ".join(cols[t].get(k, "n/a") for t in args.tags) + " |")


if __name__ == "__main__":
    main()
