"""Evaluate extracted results against the human-curated MetaLead gold (Timmer, Bölücü, Wan; EACL 2026; github.com/RoelTim/metalead, 43 NLP papers,
3,568 annotated results with task, train / test data set, metric, score).

  CGRAG_OVERLAY=config/bench_metalead.toml python scripts/eval_metalead.py                  # the pipeline's stored profiles, raw and repaired
  python scripts/eval_metalead.py --pred data/bench/metalead/baseline_gemma.json --name baseline    # any extractor that writes {paper: [ {task,dataset,metric,value}, ...]}

This is NOT MetaLead's own scorer (exact set match of normalised tuples including the experiment type, with names normalised by GPT-4.1). It is a looser,
fully deterministic reading made for a system whose names are free text, reported in three layers so each can be judged on its own:

  coverage   share of gold results whose number occurs in a prediction of the same paper (tolerance 0.005, also x100 / 1/100): did the extractor find
             the result at all?
  agreement  among the covered results, share whose predicted data set / metric / task name agrees with the gold name (the best prediction among those
             with the same number is used, so a repeated number is not punished): did it describe the result correctly?
  precision  share of predictions whose number is in the paper's gold (appendices and figures are outside MetaLead's scope, so this is a lower bound).

Names agree when their normalised token sets contain one another (language codes expanded, "WMT'14 EN-FR" = "WMT'14 English-French"), metrics through a
small alias table, tasks through the task families of src/cgrag/knowledge/benchmarks.py. The aliases were written from the corpus of the 28 development papers
and the gold's metric list, before any score was computed; they are listed in this file so a reviewer can read them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cgrag.knowledge.benchmarks import CODE_NAMES, families  # noqa: E402

GOLD = ROOT / "data/bench/metalead/annotations/metalead_annotations.json"

_STOP = {"the", "of", "and", "a", "an", "dataset", "corpus", "task", "set", "data", "benchmark", "test", "dev", "development", "validation", "train", "training"}
_METRIC_ALIASES = {
    "f1": "f1", "fscore": "f1", "f1score": "f1", "fmeasure": "f1", "microf1": "f1", "macrof1": "f1", "f": "f1", "fs": "f1",
    "acc": "accuracy", "accuracy": "accuracy", "overallaccuracy": "accuracy", "sentaccuracy": "accuracy", "sentenceaccuracy": "accuracy",
    "em": "em", "exactmatch": "em", "exactmatchem": "em", "rogue": "rouge", "rogue1": "rouge1", "rogue2": "rouge2", "roguel": "rougel",
    "r1": "rouge1", "r2": "rouge2", "rl": "rougel", "rougel": "rougel", "ppl": "perplexity", "perplexity": "perplexity",
    "err": "errorrate", "errorrate": "errorrate", "wer": "errorrate", "mcc": "matthews", "matthewscorrelationcoefficientmcc": "matthews",
    "matthewscorrelation": "matthews", "matthewscorrelationcoefficient": "matthews", "spearman": "spearman", "spearmancorrelation": "spearman",
    "spearmansrankcorrelation": "spearman", "pearson": "pearson", "pearsoncorrelation": "pearson", "bleu4": "bleu", "bleu": "bleu", "las": "las",
    "labeledattachmentscore": "las", "uas": "uas", "unlabeledattachmentscore": "uas", "p": "precision", "r": "recall",
}


def _clean(name: str | None) -> str:
    """Lower case, ASCII only, a four-digit year written short ("2003" -> "03", so CoNLL-2003 = CoNLL03 = CoNLL'03)."""
    s = re.sub(r"[^\x00-\x7f]", " ", name or "").lower()
    s = re.sub(r"(?<![0-9])(?:19|20)(\d\d)(?![0-9])", r"\1", s)
    s = re.sub(r"\bv(\d)", r"\1", s)                    # "v5" = "5.0"
    return re.sub(r"(\d)\.0\b", r"\1", s)


def _stem(t: str) -> str:
    return t[:6] if len(t) > 6 else t                       # "dialogue" = "dialog", "newstest13" = "newste"


def _tokens(name: str | None) -> set[str]:
    s = re.sub(r"[()\[\]/_,;:]", " ", _clean(name))
    out = set()
    for t in re.findall(r"[a-z0-9]+", s):
        if t in _STOP:
            continue
        out.add(_stem(CODE_NAMES.get(t, t).lower() if len(t) in (2, 3) and t in CODE_NAMES else t))
    out |= {t[-2:] for t in out if len(t) > 3 and t[-2:].isdigit() and not t.isdigit()}      # "newstest13" also carries "13"
    return out


def _compact(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", _clean(name))


def _initials(name: str | None) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", re.sub(r"[()\[\]/_,;:]", " ", _clean(name))) if w not in _STOP]
    return "".join(w[0] for w in words)


def names_agree(a: str | None, b: str | None) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        return True
    ca, cb = _compact(a), _compact(b)
    if len(min(ca, cb, key=len)) >= 4 and (ca in cb or cb in ca):
        return True
    ia, ib = _initials(a), _initials(b)
    return (len(ca) >= 3 and ca == ib) or (len(cb) >= 3 and cb == ia)       # ATIS = Airline Travel Information Systems


def metric_key(m: str | None) -> str:
    c = _compact(m)
    c = re.sub(r"^(?:top|avg|average|mean|macro|micro|overall)", "", c) or c
    return _METRIC_ALIASES.get(c, c)


def metrics_agree(a: str | None, b: str | None) -> bool:
    ka, kb = metric_key(a), metric_key(b)
    if not ka or not kb:
        return False
    return ka == kb or (min(len(ka), len(kb)) >= 3 and (ka.startswith(kb) or kb.startswith(ka)))


def tasks_agree(a: str | None, b: str | None) -> bool:
    fa, fb = families(a), families(b)
    if fa and fb:
        return bool(fa & fb) or names_agree(a, b)
    return names_agree(a, b)


def values_match(a: float, b: float) -> bool:
    return any(abs(x - y) <= 0.0051 for x, y in ((a, b), (a * 100, b), (a, b * 100)))


def score_paper(gold: list[dict], preds: list[dict]) -> dict:
    cov = agree_d = agree_m = agree_t = agree_all = 0
    matched_pred = set()
    for g in gold:
        cands = [(i, p) for i, p in enumerate(preds) if p.get("value") is not None and values_match(float(g["Result"]), float(p["value"]))]
        if not cands:
            continue
        cov += 1

        def fit(c):
            _, p = c
            return (names_agree(p.get("dataset"), g.get("Test-Dataset")), metrics_agree(p.get("metric"), g.get("Metric")),
                    tasks_agree(p.get("task"), g.get("Task")))
        best = max(cands, key=lambda c: sum(fit(c)))
        d, m, t = fit(best)
        agree_d += d
        agree_m += m
        agree_t += t
        agree_all += d and m and t
        matched_pred.update(i for i, _ in cands)
    gold_vals = [float(g["Result"]) for g in gold]
    prec_hits = sum(1 for p in preds if p.get("value") is not None and any(values_match(float(p["value"]), gv) for gv in gold_vals))
    return {"gold": len(gold), "covered": cov, "dataset": agree_d, "metric": agree_m, "task": agree_t, "all": agree_all,
            "preds": len(preds), "pred_in_gold": prec_hits}


def aggregate(per_paper: dict[str, dict]) -> dict:
    tot = {k: sum(r[k] for r in per_paper.values()) for k in ("gold", "covered", "dataset", "metric", "task", "all", "preds", "pred_in_gold")}
    pct = lambda a, b: round(100 * a / b, 1) if b else 0.0   # noqa: E731
    macro_cov = [100 * r["covered"] / r["gold"] for r in per_paper.values() if r["gold"]]
    return {
        **tot,
        "coverage_pct": pct(tot["covered"], tot["gold"]),
        "macro_coverage_pct": round(sum(macro_cov) / len(macro_cov), 1) if macro_cov else 0.0,
        "dataset_agree_pct": pct(tot["dataset"], tot["covered"]), "metric_agree_pct": pct(tot["metric"], tot["covered"]),
        "task_agree_pct": pct(tot["task"], tot["covered"]), "all_three_agree_pct": pct(tot["all"], tot["covered"]),
        "precision_pct": pct(tot["pred_in_gold"], tot["preds"]),
        "end_to_end_pct": pct(tot["all"], tot["gold"]),          # covered AND all three names agree, over every gold result
    }


def load_pipeline(paper_ids: list[str]) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict]:
    from cgrag.ingestion.repair import repair_paper
    from cgrag.stores.profile_store import ProfileStore
    from cgrag.stores.vector_store import VectorStore
    store, vs = ProfileStore(repaired=False), VectorStore()      # the RAW extraction: the repair is applied (and scored) on top
    raw, rep, rules = {}, {}, {}
    from collections import Counter
    total = Counter()
    for pid in paper_ids:
        profs = store.for_paper(pid)
        if not profs:
            continue
        cs = vs.paper_chunks(pid)
        fixed, counts = repair_paper(profs, {c.chunk_id: c.text for c in cs}, title=cs[0].paper_title if cs else "")
        total.update(counts)
        dump = lambda ps: [p.model_dump(include={"task", "dataset", "dataset_version", "metric", "value", "model", "setting", "language", "model_size"}) for p in ps]  # noqa: E731
        raw[pid], rep[pid] = dump(profs), dump(fixed)
    return raw, rep, dict(total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="JSON {paper_id: [ {task, dataset, metric, value, ...} ]} from another extractor")
    ap.add_argument("--name", default="pipeline")
    ap.add_argument("--out", default=str(ROOT / "eval/labels/metalead_eval.json"))
    ap.add_argument("--only", nargs="*", help="paper ids (default: every gold paper that has predictions)")
    args = ap.parse_args()

    gold_all = json.loads(Path(GOLD).read_text(encoding="utf-8"))
    gold = {k[:-4]: v["TDMs"] for k, v in gold_all.items()}
    result: dict = {"name": args.name}
    if args.pred:
        preds = json.loads(Path(args.pred).read_text(encoding="utf-8"))
        ids = [i for i in gold if i in preds and (not args.only or i in args.only)]
        per = {i: score_paper(gold[i], preds[i]) for i in ids}
        result["variants"] = {args.name: {"papers": len(ids), **aggregate(per)}}
        result["per_paper"] = {args.name: per}
    else:
        raw, rep, rules = load_pipeline([i for i in gold if not args.only or i in args.only])
        ids = [i for i in gold if i in raw]
        per_raw = {i: score_paper(gold[i], raw[i]) for i in ids}
        per_rep = {i: score_paper(gold[i], rep[i]) for i in ids}
        result["variants"] = {"stored": {"papers": len(ids), **aggregate(per_raw)}, "repaired": {"papers": len(ids), **aggregate(per_rep)}}
        result["repair_rules_fired"] = rules
        result["per_paper"] = {"stored": per_raw, "repaired": per_rep}
        missing = [i for i in gold if i not in raw]
        result["gold_papers_without_profiles"] = missing
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, agg in result["variants"].items():
        print(f"\n{name}: {agg['papers']} papers, {agg['gold']} gold results")
        for k in ("coverage_pct", "macro_coverage_pct", "dataset_agree_pct", "metric_agree_pct", "task_agree_pct", "all_three_agree_pct", "end_to_end_pct", "precision_pct"):
            print(f"  {k:<22} {agg[k]}")
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
