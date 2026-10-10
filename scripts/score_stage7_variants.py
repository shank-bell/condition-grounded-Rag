"""Stage 7 variants on both pair sets (10 Oct 2026): replay `classify` on the stored profiles of the 100 labelled pairs under different switch settings.

  python scripts/score_stage7_variants.py        # needs the private keys and label workbooks; writes eval/labels/stage7_variants.json (aggregates only)

Four sets of 50 pairs: the first 50 (tuned on for fix A), the held-out 50 (scored once for the frozen commit), a FRESH 50 (10 Oct, built after policy B was chosen;
its error types were read while table grounding was designed) and FRESH-2 (10 Oct, built and labelled AFTER table grounding was frozen, blind to the system's verdicts, scored once: the clean test). Policy B (`one_sided_conditions_explain`: a condition recorded for only one of the
two papers is the probable explanation) was found by reading the mistakes of the frozen Stage 7 on the first two sets, so the numbers on those two are
DEVELOPMENT numbers; the fresh set is the clean test. Baselines (plain NLI, Gemma 12B with the conflict-taxonomy prompt, "always EXPLAINED") are scored
on the same pairs where their private files exist.
Only the pair profiles are re-classified (same pairs, same gold); nothing is re-retrieved. The replay reproduces the stored verdicts of the frozen
commit and of fix A (checked pair by pair, mismatches are printed).
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

from cgrag.config import ContradictionConfig, get_settings
from cgrag.labelling import score as S
from cgrag.labelling.sampling import load_chunk_index
from cgrag.pipeline.contradiction import classify
from cgrag.pipeline.grounding import ground
from cgrag.stores.profile_store import ProfileStore

DONE = Path("data/labelling/done")
PRIVATE = Path("data/labelling/private")
REPAIRED = os.environ.get("PROFILE_REPAIR", "off") == "on"       # PROFILE_REPAIR=on: read the pair profiles through the repair overlay (11 Oct); default = the raw store as in the frozen results
OUT = Path("eval/labels/stage7_variants_repaired.json" if REPAIRED else "eval/labels/stage7_variants.json")
FROZEN = dict(human_rows_not_comparable=False, two_metric_tasks_no_genuine=False, mnli_split_tags=False, one_sided_conditions_explain=False, table_grounding=False)
VARIANTS = {
    "frozen (every switch off)": FROZEN,
    "fix A (before table grounding)": dict(one_sided_conditions_explain=False, table_grounding=False),
    "policy B alone": dict(human_rows_not_comparable=False, two_metric_tasks_no_genuine=False, mnli_split_tags=False, one_sided_conditions_explain=True, table_grounding=False),
    "fix A + policy B": dict(one_sided_conditions_explain=True, table_grounding=False),
    "fix A + table grounding (shipped 10 Oct)": dict(table_grounding=True, one_sided_conditions_explain=False),
    "fix A + grounding + policy B (shipped from 11 Oct)": dict(table_grounding=True, one_sided_conditions_explain=True),
}


def load_gold(first: str, second: str, third: str) -> dict:
    a, _ = S.parse_job_c(DONE / first)
    b, _ = S.parse_job_c(DONE / second)
    c, _ = S.parse_job_c(DONE / third)
    return S.gold_job_c(a, b, c)


def main() -> None:
    fresh_first, _ = S.parse_job_c(DONE / "C_fresh_pairs_first_AIannotated-fresh-first_DONE.xlsx")
    fresh_gold = {p: {"verdict": v["verdict"], "differs": v["differs"], "extraction_error": v["extraction_error"], "settled_by_third": False} for p, v in fresh_first.items()}
    fresh2_first, _ = S.parse_job_c(DONE / "C_fresh2_pairs_first_AIannotated-fresh2-first_DONE.xlsx")
    fresh2_gold = {p: {"verdict": v["verdict"], "differs": v["differs"], "extraction_error": v["extraction_error"], "settled_by_third": False} for p, v in fresh2_first.items()}
    fresh3_first, _ = S.parse_job_c(DONE / "C_fresh3_pairs_first_AIannotated-fresh3-first_DONE.xlsx")
    fresh3_gold = {p: {"verdict": v["verdict"], "differs": v["differs"], "extraction_error": v["extraction_error"], "settled_by_third": False} for p, v in fresh3_first.items()}
    sets = {
        "first 50 pairs": (load_gold("C_result_pairs_Aditya_AIannotated-pass1_DONE.xlsx", "C_result_pairs_Tarun_AIannotated-gemini31pro_DONE.xlsx",
                                     "C_result_pairs_Tarun_AIannotated-third_DONE.xlsx"), json.loads((PRIVATE / "job_C_key.json").read_text(encoding="utf-8")),
                           "system_verdict"),
        "held-out 50 pairs": (load_gold("C_heldout_pairs_first_AIannotated-heldout-first_DONE.xlsx", "C_heldout_pairs_second_AIannotated-heldout-gemini_DONE.xlsx",
                                        "C_heldout_pairs_first_AIannotated-heldout-third_DONE.xlsx"),
                              json.loads((PRIVATE / "job_C_heldout_key.json").read_text(encoding="utf-8")), "system_verdict_frozen"),
        "fresh 50 pairs (clean test)": (fresh_gold, json.loads((PRIVATE / "job_C_fresh_key.json").read_text(encoding="utf-8")), "system_verdict_frozen"),
        "fresh-2 50 pairs (final clean test)": (fresh2_gold, json.loads((PRIVATE / "job_C_fresh2_key.json").read_text(encoding="utf-8")), "system_verdict_frozen"),
        "fresh-3 50 pairs (clean test of the repaired store, 11 Oct)": (fresh3_gold, json.loads((PRIVATE / "job_C_fresh3_key.json").read_text(encoding="utf-8")),
                                                                         "system_verdict_frozen"),
    }
    baseline_files = {"first 50 pairs": PRIVATE / "job_C_baselines.json", "held-out 50 pairs": PRIVATE / "job_C_heldout_baselines.json",
                      "fresh 50 pairs (clean test)": PRIVATE / "job_C_fresh_baselines.json",
                      "fresh-2 50 pairs (final clean test)": PRIVATE / "job_C_fresh2_baselines.json",
                      "fresh-3 50 pairs (clean test of the repaired store, 11 Oct)": PRIVATE / "job_C_fresh3_baselines.json"}
    chunk_text = load_chunk_index(get_settings().paths.index_dir / "chunks.jsonl").texts
    con = sqlite3.connect(get_settings().paths.profile_db)
    con.row_factory = sqlite3.Row
    store = ProfileStore(repaired=REPAIRED)
    print("profiles read through the repair overlay:", REPAIRED)

    def profile(pid: str):
        return store._row(con.execute("select * from profiles where profile_id=?", (pid,)).fetchone())

    result: dict = {"note": "development numbers (policy B was found by reading the mistakes of both sets); aggregates only", "sets": {}, "pooled_100_pairs": {}}
    pooled_gold: dict = {}
    pooled_preds: dict[str, dict] = {name: {} for name in VARIANTS}
    all_gold: dict = {}                                   # all three sets (150 pairs); ids differ by prefix (P / H / F)
    all_preds: dict[str, dict] = {name: {} for name in VARIANTS}
    one_sided: dict = {}
    for set_name, (gold, key, frozen_field) in sets.items():
        if not set_name.startswith("fresh"):
            pooled_gold.update(gold)                      # the pooled-100 numbers are over the first two sets only
        all_gold.update(gold)
        reason_field = "reason_frozen" if "reason_frozen" in next(iter(key.values())) else "reason"
        one = Counter(gold[p]["verdict"] for p in gold if "recorded for only one of the two papers" in key[p][reason_field])
        one_sided[set_name] = {"pairs_where_the_frozen_rule_says_recorded_for_only_one_paper": sum(one.values()), "gold": dict(one)}
        result["sets"][set_name] = {}
        for name, switches in VARIANTS.items():
            cfg = ContradictionConfig(**switches)
            preds, mismatch = {}, 0
            for pid in gold:
                a, b = profile(key[pid]["profile_a"]), profile(key[pid]["profile_b"])
                ga, gb = (ground(a, chunk_text.get(a.chunk_id, "")), ground(b, chunk_text.get(b.chunk_id, ""))) if cfg.table_grounding else (None, None)
                verdict, differing, _ = classify(a, b, cfg, ga, gb)
                preds[pid] = {"system_verdict": verdict, "system_differing": differing}
                if name.startswith("frozen") and verdict != key[pid][frozen_field]:
                    mismatch += 1
            all_preds[name].update(preds)
            if not set_name.startswith("fresh"):
                pooled_preds[name].update(preds)
            sc = S.score_system_c(gold, preds)
            gen = sum(p["system_verdict"] == "GENUINE" for p in preds.values())
            nc_gold = [p for p in gold if gold[p]["verdict"] == "NOT COMPARABLE"]
            nc_found = sum(S.SYSTEM_VERDICT[preds[p]["system_verdict"]] == "NOT COMPARABLE" for p in nc_gold)
            result["sets"][set_name][name] = {"accuracy": sc["accuracy"], "right": round(sc["accuracy"] * len(gold)), "pairs": len(gold), "macro_f1": sc["macro_f1"],
                                              "genuine_calls": gen, "not_comparable_found": f"{nc_found}/{len(nc_gold)}", "verdicts": dict(Counter(p["system_verdict"] for p in preds.values())),
                                              "condition_attribution_merged_exact": sc["condition_attribution_merged"]["exact_match"],
                                              "condition_attribution_merged_pairs": sc["condition_attribution_merged"]["pairs"],
                                              "condition_attribution_merged_jaccard": sc["condition_attribution_merged"]["mean_jaccard"],
                                              "per_class_f1": {c: v["f1"] for c, v in sc["per_class"].items()}}
            if name.startswith("frozen"):
                print(f"[{set_name}] replay of the frozen rules differs from the stored frozen verdicts on {mismatch} pair(s)")
    for set_name, (gold, key, _) in sets.items():
        path = baseline_files[set_name]
        if not path.exists():
            continue
        base = json.loads(path.read_text(encoding="utf-8"))["systems"]
        extra = {"plain NLI": base["plain_nli"], "Gemma 12B (taxonomy prompt)": base["llm_taxonomy"],
                 "always EXPLAINED": {p: {"system_verdict": "EXPLAINED", "system_differing": []} for p in gold}}
        for name, preds in extra.items():
            preds = {p: {"system_verdict": preds[p]["system_verdict"], "system_differing": preds[p].get("system_differing", [])} for p in gold}
            sc = S.score_system_c(gold, preds)
            nc_gold = [p for p in gold if gold[p]["verdict"] == "NOT COMPARABLE"]
            nc_found = sum(S.SYSTEM_VERDICT[preds[p]["system_verdict"]] == "NOT COMPARABLE" for p in nc_gold)
            result["sets"][set_name][name] = {"accuracy": sc["accuracy"], "right": round(sc["accuracy"] * len(gold)), "pairs": len(gold), "macro_f1": sc["macro_f1"],
                                              "genuine_calls": sum(p["system_verdict"] == "GENUINE" for p in preds.values()), "not_comparable_found": f"{nc_found}/{len(nc_gold)}",
                                              "verdicts": dict(Counter(p["system_verdict"] for p in preds.values())),
                                              "condition_attribution_merged_exact": sc["condition_attribution_merged"]["exact_match"],
                                              "condition_attribution_merged_pairs": sc["condition_attribution_merged"]["pairs"],
                                              "condition_attribution_merged_jaccard": sc["condition_attribution_merged"]["mean_jaccard"],
                                              "per_class_f1": {c: v["f1"] for c, v in sc["per_class"].items()}}
    result["one_sided_pairs"] = one_sided
    from cgrag.evaluation.answers import sign_test
    result["paired"] = {}
    for set_name, (gold, key, _) in sets.items():
        right = {}
        for name in ("fix A (before table grounding)", "fix A + table grounding (shipped 10 Oct)", "fix A + grounding + policy B (shipped from 11 Oct)"):
            cfg = ContradictionConfig(**VARIANTS[name])
            for pid in gold:
                a, b = profile(key[pid]["profile_a"]), profile(key[pid]["profile_b"])
                ga, gb = (ground(a, chunk_text.get(a.chunk_id, "")), ground(b, chunk_text.get(b.chunk_id, ""))) if cfg.table_grounding else (None, None)
                right.setdefault(name, {})[pid] = S.SYSTEM_VERDICT[classify(a, b, cfg, ga, gb)[0]] == gold[pid]["verdict"]
        up = sum(right["fix A + table grounding (shipped 10 Oct)"][p] and not right["fix A (before table grounding)"][p] for p in gold)
        down = sum(right["fix A (before table grounding)"][p] and not right["fix A + table grounding (shipped 10 Oct)"][p] for p in gold)
        always_up = sum(right["fix A + table grounding (shipped 10 Oct)"][p] and gold[p]["verdict"] != "EXPLAINED" for p in gold)
        always_down = sum((not right["fix A + table grounding (shipped 10 Oct)"][p]) and gold[p]["verdict"] == "EXPLAINED" for p in gold)
        b_up = sum(right["fix A + grounding + policy B (shipped from 11 Oct)"][p] and not right["fix A + table grounding (shipped 10 Oct)"][p] for p in gold)
        b_down = sum(right["fix A + table grounding (shipped 10 Oct)"][p] and not right["fix A + grounding + policy B (shipped from 11 Oct)"][p] for p in gold)
        bal_up = sum(right["fix A + grounding + policy B (shipped from 11 Oct)"][p] and gold[p]["verdict"] != "EXPLAINED" for p in gold)
        bal_down = sum((not right["fix A + grounding + policy B (shipped from 11 Oct)"][p]) and gold[p]["verdict"] == "EXPLAINED" for p in gold)
        result["paired"][set_name] = {"grounding_vs_fixA": {"gained": up, "lost": down, "sign_test_p": round(sign_test(up, down), 4)},
                                      "grounding_plus_B_vs_grounding": {"gained": b_up, "lost": b_down, "sign_test_p": round(sign_test(b_up, b_down), 4)},
                                      "grounding_plus_B_vs_always_explained": {"gained": bal_up, "lost": bal_down, "sign_test_p": round(sign_test(bal_up, bal_down), 4)},
                                      "grounding_vs_always_explained": {"gained": always_up, "lost": always_down, "sign_test_p": round(sign_test(always_up, always_down), 4)}}
    result["gold_counts"] = {n: dict(Counter(g["verdict"] for g in gold.values())) for n, (gold, _, _) in sets.items()}
    # pooled over the 100 pairs
    genuine_gold = {p for p, g in pooled_gold.items() if g["verdict"] == "GENUINE"}
    for name in VARIANTS:
        preds = pooled_preds[name]
        sc = S.score_system_c(pooled_gold, preds)
        wrong_alarms = sum(preds[p]["system_verdict"] == "GENUINE" for p in pooled_gold if p not in genuine_gold)
        found = sum(preds[p]["system_verdict"] == "GENUINE" for p in genuine_gold)
        result["pooled_100_pairs"][name] = {"accuracy": sc["accuracy"], "right": round(sc["accuracy"] * len(pooled_gold)), "macro_f1": sc["macro_f1"],
                                            "false_genuine_calls": wrong_alarms, "of_not_genuine": len(pooled_gold) - len(genuine_gold),
                                            "real_contradictions_found": f"{found}/{len(genuine_gold)}",
                                            "condition_attribution_merged_exact": sc["condition_attribution_merged"]["exact_match"],
                                            "condition_attribution_merged_pairs": sc["condition_attribution_merged"]["pairs"],
                                            "verdicts": dict(Counter(p["system_verdict"] for p in preds.values()))}
    result["pooled_150_pairs"] = {}
    nc_all = [p for p, g in all_gold.items() if g["verdict"] == "NOT COMPARABLE"]
    for name in VARIANTS:
        preds = all_preds[name]
        sc = S.score_system_c(all_gold, preds)
        found = sum(S.SYSTEM_VERDICT[preds[p]["system_verdict"]] == "NOT COMPARABLE" for p in nc_all)
        result["pooled_150_pairs"][name] = {"accuracy": sc["accuracy"], "right": round(sc["accuracy"] * len(all_gold)), "macro_f1": sc["macro_f1"],
                                            "not_comparable_found": f"{found}/{len(nc_all)}", "verdicts": dict(Counter(p["system_verdict"] for p in preds.values()))}
    result["pooled_150_pairs"]["always EXPLAINED"] = {"right": sum(g["verdict"] == "EXPLAINED" for g in all_gold.values()), "of": len(all_gold)}
    OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for set_name, block in result["sets"].items():
        print(f"\n{set_name}")
        for name, r in block.items():
            print(f"  {name:28s} right {r['right']:2d}/{r['pairs']} = {100 * r['accuracy']:.0f}%  macro-F1 {100 * (r['macro_f1'] or 0):.0f}  GENUINE calls {r['genuine_calls']}  "
                  f"NC found {r['not_comparable_found']}  naming (merged) {r['condition_attribution_merged_exact']} of {r['condition_attribution_merged_pairs']}  {r['verdicts']}")
    print("\npooled over the 100 pairs")
    for name, r in result["pooled_100_pairs"].items():
        print(f"  {name:28s} right {r['right']:3d}/100  macro-F1 {100 * (r['macro_f1'] or 0):.0f}  false GENUINE {r['false_genuine_calls']}/{r['of_not_genuine']}  "
              f"real found {r['real_contradictions_found']}  naming {r['condition_attribution_merged_exact']} of {r['condition_attribution_merged_pairs']}")
    print("\none-sided pairs (the frozen rule says 'recorded for only one of the two papers'):", json.dumps(result["one_sided_pairs"]))
    print("\npooled over all four sets (200 pairs; three of them used for development)")
    for name, r in result["pooled_150_pairs"].items():
        print(f"  {name:28s} {json.dumps(r)}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
