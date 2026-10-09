"""Score Stage 7 ONCE on the held-out result pairs (job C, ids H01..H50) against the settled AI-annotated key.

  python scripts/score_heldout_c.py            # reads the private key, the three label workbooks and the baselines; writes eval/labels/job_C_heldout_scores.json

The held-out pairs were built (scripts/make_heldout_pairs.py) and labelled blind to the system AFTER fix A of Stage 7 was frozen; nothing may be
changed in Stage 7 after this has been looked at, or the numbers stop being held-out. Systems scored on the same settled key:
  stage7 frozen   Stage 7 with every fix-A switch off (what the frozen commit said)
  stage7 fix A    Stage 7 as it is now
  plain NLI       DeBERTa NLI on the two statements
  Gemma 12B       the local LLM with the conflict-taxonomy prompt (same prompt as for the first 50 pairs)
  always EXPLAINED  a reference that is right whenever the key says EXPLAINED
Only aggregates are written (no labels, no verdicts per pair), so the output is safe to publish.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from cgrag.evaluation.answers import sign_test
from cgrag.evaluation.freeze import provenance
from cgrag.labelling import score as S

DONE = Path("data/labelling/done")
PRIVATE = Path("data/labelling/private")
OUT = Path("eval/labels/job_C_heldout_scores.json")
NAME = {"EXPLAINED": "EXPLAINED", "GENUINE": "GENUINE", "NOT_COMPARABLE": "NOT COMPARABLE"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def pooled(held_gold: dict, held_key: dict, held_base: dict) -> dict:
    """All 100 pairs: the first 50 (frozen Stage 7 verdicts of the original key, the settled first-set key, the frozen baselines) plus the held-out 50
    (Stage 7 with every fix-A switch off). The question: how often does a system call a pair a real contradiction (GENUINE) when the key says it is not?"""
    first, _ = S.parse_job_c(DONE / "C_result_pairs_Aditya_AIannotated-pass1_DONE.xlsx")
    second, _ = S.parse_job_c(DONE / "C_result_pairs_Tarun_AIannotated-gemini31pro_DONE.xlsx")
    third, _ = S.parse_job_c(DONE / "C_result_pairs_Tarun_AIannotated-third_DONE.xlsx")
    gold1 = S.gold_job_c(first, second, third)
    key1 = json.loads((PRIVATE / "job_C_key.json").read_text(encoding="utf-8"))
    base1 = json.loads((PRIVATE / "job_C_baselines.json").read_text(encoding="utf-8"))["systems"]
    rows = [(g["verdict"], S.SYSTEM_VERDICT[key1[p]["system_verdict"]], S.SYSTEM_VERDICT[base1["llm_taxonomy"][p]["system_verdict"]],
             S.SYSTEM_VERDICT[base1["plain_nli"][p]["system_verdict"]]) for p, g in gold1.items()]
    rows += [(g["verdict"], S.SYSTEM_VERDICT[held_key[p]["system_verdict_frozen"]], S.SYSTEM_VERDICT[held_base["llm_taxonomy"][p]["system_verdict"]],
              S.SYSTEM_VERDICT[held_base["plain_nli"][p]["system_verdict"]]) for p, g in held_gold.items()]
    non = [r for r in rows if r[0] != "GENUINE"]
    gen = [r for r in rows if r[0] == "GENUINE"]
    out: dict = {"pairs": len(rows), "genuine_in_key": len(gen), "not_genuine_in_key": len(non), "systems": {}}
    for name, i in (("stage7 (fix A off)", 1), ("Gemma 12B (taxonomy prompt)", 2), ("plain NLI", 3)):
        fc, tc = sum(r[i] == "GENUINE" for r in non), sum(r[i] == "GENUINE" for r in gen)
        acc = sum(r[i] == r[0] for r in rows)
        out["systems"][name] = {"false_genuine_calls": fc, "of_not_genuine": len(non), "false_genuine_rate": round(fc / len(non), 4), "false_genuine_ci95": wilson(fc, len(non)),
                                "real_contradictions_found": f"{tc}/{len(gen)}", "right": acc, "accuracy": round(acc / len(rows), 4), "accuracy_ci95": wilson(acc, len(rows))}
    only_g = sum(r[1] != "GENUINE" and r[2] == "GENUINE" for r in non)
    only_s = sum(r[1] == "GENUINE" and r[2] != "GENUINE" for r in non)
    out["false_genuine_stage7_vs_gemma"] = {"only_gemma_cries_wolf": only_g, "only_stage7_cries_wolf": only_s, "sign_test_p": round(sign_test(only_g, only_s), 4)}
    a7 = sum(r[1] == r[0] and r[2] != r[0] for r in rows)
    ag = sum(r[2] == r[0] and r[1] != r[0] for r in rows)
    out["accuracy_stage7_vs_gemma"] = {"only_stage7_right": a7, "only_gemma_right": ag, "sign_test_p": round(sign_test(a7, ag), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, default=PRIVATE / "job_C_heldout_key.json")
    ap.add_argument("--baselines", type=Path, default=PRIVATE / "job_C_heldout_baselines.json")
    ap.add_argument("--first", type=Path, default=DONE / "C_heldout_pairs_first_AIannotated-heldout-first_DONE.xlsx")
    ap.add_argument("--second", type=Path, default=DONE / "C_heldout_pairs_second_AIannotated-heldout-gemini_DONE.xlsx")
    ap.add_argument("--third", type=Path, default=DONE / "C_heldout_pairs_first_AIannotated-heldout-third_DONE.xlsx")
    args = ap.parse_args()

    first, p1 = S.parse_job_c(args.first)
    second, p2 = S.parse_job_c(args.second)
    third, p3 = S.parse_job_c(args.third)
    gold = S.gold_job_c(first, second, third)
    for pid in third:                                   # where the labellers disagreed on the flag, the third look decides it
        gold[pid]["extraction_error"] = bool(third[pid]["extraction_error"])
    low = {p for p in gold if str((third.get(p) or first[p]).get("confidence", "")).lower().startswith("low")}
    xerr = {p for p, g in gold.items() if g["extraction_error"]}
    key = json.loads(args.key.read_text(encoding="utf-8"))
    base = json.loads(args.baselines.read_text(encoding="utf-8"))["systems"]
    systems = {
        "stage7 frozen (fix A off)": {p: {"system_verdict": k["system_verdict_frozen"], "system_differing": k["system_differing_frozen"]} for p, k in key.items()},
        "stage7 fix A": {p: {"system_verdict": k["system_verdict"], "system_differing": k["system_differing"]} for p, k in key.items()},
        "plain NLI": base["plain_nli"],
        "Gemma 12B (taxonomy prompt)": base["llm_taxonomy"],
        "always EXPLAINED (reference)": {p: {"system_verdict": "EXPLAINED", "system_differing": []} for p in key},
    }
    agreement = S.agreement_job_c(first, second)
    result = {"pairs": len(gold), "key_verdicts": dict(Counter(g["verdict"] for g in gold.values())), "labellers": "AI: Claude (first pass, third look), Gemini (second pass); no human",
              "agreement_first_vs_second": {k: v for k, v in agreement.items() if k != "disagreements"} | {"n_disagreements": len(agreement["disagreements"])},
              "extraction_errors_in_key": len(xerr), "low_confidence_in_key": len(low), "problems": p1 + p2 + p3, "systems": {}}
    print(f"held-out key: {result['key_verdicts']} | agreement first vs second {agreement['percent_agreement']:.0%}, kappa {agreement['kappa']} | extraction errors {len(xerr)}, low confidence {len(low)}")
    print(f"\n{'system':<30}{'right':>7}{'acc':>7}{'95% CI':>15}{'macroF1':>9}{'GENUINE':>9}{'NC found':>10}{'acc w/o low':>13}{'acc w/o xerr':>14}")
    correct: dict[str, dict[str, bool]] = {}
    n_nc = sum(g["verdict"] == "NOT COMPARABLE" for g in gold.values())
    for name, preds in systems.items():
        sc = S.score_system_c(gold, preds)
        ok = {p: S.SYSTEM_VERDICT[preds[p]["system_verdict"]] == gold[p]["verdict"] for p in gold}
        correct[name] = ok
        k = sum(ok.values())
        keep_low = [p for p in gold if p not in low]
        keep_x = [p for p in gold if p not in xerr]
        n_gen = sum(S.SYSTEM_VERDICT[preds[p]["system_verdict"]] == "GENUINE" for p in gold)
        nc_found = sum(ok[p] and gold[p]["verdict"] == "NOT COMPARABLE" for p in gold)
        ci = wilson(k, len(gold))
        a_low = sum(ok[p] for p in keep_low) / len(keep_low)
        a_x = sum(ok[p] for p in keep_x) / len(keep_x)
        print(f"{name:<30}{k:>4}/{len(gold)}{100 * k / len(gold):>6.1f}%  {100 * ci[0]:>5.1f}-{100 * ci[1]:>5.1f}%{100 * (sc['macro_f1'] or 0):>8.1f}{n_gen:>9}{nc_found:>6}/{n_nc}{100 * a_low:>12.1f}%{100 * a_x:>13.1f}%")
        result["systems"][name] = {**sc, "right": k, "accuracy_ci95": ci, "genuine_calls": n_gen, "not_comparable_found": f"{nc_found}/{n_nc}",
                                   "accuracy_without_low_confidence_pairs": round(a_low, 4), "pairs_without_low_confidence": len(keep_low),
                                   "accuracy_without_extraction_error_pairs": round(a_x, 4), "pairs_without_extraction_error": len(keep_x)}
    a, b = correct["stage7 fix A"], correct["stage7 frozen (fix A off)"]
    only_a = sum(a[p] and not b[p] for p in gold)
    only_b = sum(b[p] and not a[p] for p in gold)
    result["fix_A_vs_frozen"] = {"only_fix_A_right": only_a, "only_frozen_right": only_b, "sign_test_p": round(sign_test(only_a, only_b), 4)}
    g, f = correct["stage7 fix A"], correct["Gemma 12B (taxonomy prompt)"]
    og, of = sum(g[p] and not f[p] for p in gold), sum(f[p] and not g[p] for p in gold)
    result["fix_A_vs_gemma"] = {"only_fix_A_right": og, "only_gemma_right": of, "sign_test_p": round(sign_test(og, of), 4)}
    print(f"\nfix A vs frozen: only fix A right {only_a}, only frozen right {only_b}, sign test p = {result['fix_A_vs_frozen']['sign_test_p']}")
    print(f"fix A vs Gemma 12B: only fix A right {og}, only Gemma right {of}, sign test p = {result['fix_A_vs_gemma']['sign_test_p']}")
    result["pooled_100_pairs"] = pooled(gold, key, base)
    for name, s in result["pooled_100_pairs"]["systems"].items():
        print(f"pooled 100 pairs, {name:<28} false 'real contradiction' calls {s['false_genuine_calls']}/{s['of_not_genuine']} = {100 * s['false_genuine_rate']:.1f}% "
              f"(CI {100 * s['false_genuine_ci95'][0]:.1f}-{100 * s['false_genuine_ci95'][1]:.1f}%), real contradictions found {s['real_contradictions_found']}, accuracy {s['right']}/100")
    print("pooled: false alarms Stage 7 vs Gemma", result["pooled_100_pairs"]["false_genuine_stage7_vs_gemma"], "| accuracy", result["pooled_100_pairs"]["accuracy_stage7_vs_gemma"])
    result["freeze_record"] = provenance()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
