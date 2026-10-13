"""Evaluate the stored profiles of the 28 development papers against an INDEPENDENT answer key: the community-curated leaderboards of Papers with Code
(the last public snapshot, 28 July 2025, CC-BY-SA-4.0, archived at huggingface.co/datasets/pwc-archive/files; `jul-28-evaluation-tables.json.gz`, 21.8 MB).

  python scripts/eval_pwc_gold.py                       # all matched papers, stored (raw) and repaired profiles
  python scripts/eval_pwc_gold.py --split dev --show 40 # print disagreements / misses of the development papers only
  python scripts/eval_pwc_gold.py --split test          # the half of the papers that was set aside by a hash of the arXiv id before the first run

Why. Every other answer key of this project was written by one AI labeller (the author of the repair rules). These leaderboards were made by the papers' authors
and the community, long before this system existed, and they cover the corpus the system actually answers from: a row is (task, data set, metric, model, number)
of one paper. They are NOT complete (only results that someone put on a leaderboard) and name things in Papers with Code's own style.

What counts. A row is attributed to the paper that introduced the MODEL, so it can be a third party's result (BERT on a stock-market data set): such a row can
not be found in the paper by anybody. A row is therefore RECOVERABLE only when its number is printed in the paper's own text (any decimal form: 0.925 or
92.5); coverage and agreement are computed on those rows. Three layers, as for MetaLead (scripts/eval_metalead.py, whose matchers are used after the
conventions of Papers with Code are mapped onto the paper's: "SQuAD1.1 dev" = SQuAD 1.1, "MultiNLI" = MNLI, "Matched" is a split, not a metric, "(BPC)"):
coverage (the number occurs among the paper's stored results), agreement of data set / metric / model among the covered results ("identity": what Stages
6 and 7 need), and the task name, reported separately because Papers with Code files a data set under a task by its own convention (MRPC under
"Semantic Textual Similarity", QQP under "Question Answering").

Use. Evaluation and error finding only: a row is never used to fill or overwrite a stored field (that would be the answer key, and a new paper has no
leaderboard row). The set-aside papers (`--split test`) are for one reading after a rule is frozen.
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_metalead as E
from cgrag.config import get_settings
from cgrag.knowledge.benchmarks import lookup
from cgrag.labelling.sampling import load_chunk_index
from cgrag.pipeline.conditions import model_family
from cgrag.stores.profile_store import ProfileStore

TABLES = ROOT / "data/bench/pwc/evaluation-tables.json.gz"
URL = "https://huggingface.co/datasets/pwc-archive/files/resolve/main/jul-28-evaluation-tables.json.gz"
NUMBER = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])")
UNSPECIFIED_METRICS = {"matched", "mismatched", "test", "dev", "average", "avg", "score", "overall", "indomain", "outofdomain", "all", "val", "validation"}
ABBREVIATIONS = {"ens": "ensemble", "sgl": "single", "ensembles": "ensemble"}


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def split_of(pid: str) -> str:
    """Deterministic, decided by the arXiv id alone (so before any result is seen): half of the papers are set aside."""
    return "test" if int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2 == 1 else "dev"


def number(text) -> float | None:
    m = re.search(r"-?\d+(?:[.,]\d+)?", str(text).replace(",", "."))
    try:
        return float(m.group(0)) if m else None
    except ValueError:
        return None


def load_rows() -> dict[str, list[dict]]:
    """Normalised paper title -> [{task, dataset, model, metric, value, raw}]."""
    if not TABLES.exists():
        TABLES.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, TABLES)
    data = json.load(gzip.open(TABLES, "rt", encoding="utf-8"))
    out: dict[str, list[dict]] = defaultdict(list)

    def walk(task: dict):
        yield task
        for sub in task.get("subtasks") or []:
            yield from walk(sub)

    for top in data:
        for task in walk(top):
            for ds in task.get("datasets") or []:
                for row in (ds.get("sota") or {}).get("rows") or []:
                    for metric, raw in (row.get("metrics") or {}).items():
                        v = number(raw)
                        if v is not None:
                            out[_norm(row.get("paper_title"))].append({"task": task["task"], "dataset": ds["dataset"], "model": row.get("model_name") or "",
                                                                        "metric": metric, "value": v, "raw": str(raw), "paper_title": row.get("paper_title")})
    return out


def rows_of(title: str, by_title: dict[str, list[dict]]) -> tuple[list[dict], str]:
    n = _norm(title)
    if n in by_title:
        return by_title[n], "exact"
    near = [k for k in by_title if len(k) > 12 and (k in n or n in k)]
    how = "contains"
    if not near:
        near = difflib.get_close_matches(n, list(by_title), n=1, cutoff=0.92)
        how = "close"
    return [r for k in near[:2] for r in by_title[k]], how


def printed_numbers(text: str) -> set[str]:
    out = set()
    for m in NUMBER.finditer(text):
        s = m.group(0)
        out.add(s)
        if "." in s:
            out.add(s.rstrip("0").rstrip("."))
    return out


def named_in_paper(dataset: str | None, text_compact: str) -> bool:
    """The paper mentions the data set (its Papers with Code name, a known alias, or a distinctive word of it): a leaderboard row for a data set that the
    paper never names was reported by a third party that used the paper's model."""
    name = canon_dataset(dataset)
    b = lookup(name)
    candidates = {_norm(name), *(_norm(a) for a in (b.aliases if b else ())), _norm(b.name) if b else ""}
    words = (_norm(t) for t in re.findall(r"[A-Za-z][A-Za-z0-9-]+", name))
    candidates |= {w for w in words if len(w) >= 5 and w not in {"dataset", "benchmark", "challenge", "language"}}
    return any(c and c in text_compact for c in candidates)


def recoverable(row: dict, numbers: set[str], text_compact: str = "") -> bool:
    """The number of the row is printed in the paper (as given, to 1 or 2 decimals, or as a fraction x100) and the paper names the data set."""
    if text_compact and not named_in_paper(row["dataset"], text_compact):
        return False
    if re.search(r"param|size|flops|speed|latency|time|memory|throughput", row["metric"], re.I):
        return False
    return bool(printed_forms(row) & numbers)


def printed_forms(row: dict) -> set[str]:
    """The ways in which the paper may print this leaderboard value (see `recoverable`)."""
    v = row["value"]
    m = re.search(r"\d+(?:\.(\d+))?", row["raw"].replace(",", "."))
    n = len(m.group(1)) if m and m.group(1) else 0
    forms = {m.group(0)} if m else {f"{v:g}"}
    if n >= 3:
        forms |= {f"{v:.1f}", f"{v:.2f}"}
    if abs(v) <= 1.0 and n >= 2:
        forms |= {f"{v * 100:.{n - 2}f}"} | ({f"{v * 100:.1f}"} if n >= 4 else set())
    forms |= {f.rstrip("0").rstrip(".") for f in forms if "." in f}
    return forms


PWC_DATASET_NAMES = {"mml": "MMLU"}


def canon_dataset(name: str | None) -> str:
    name = PWC_DATASET_NAMES.get(_norm(name), name)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", name or "")
    s = re.sub(r"\((?:[^)]*)\)", " ", s) if len(s) > 14 else s
    return re.sub(r"\s+", " ", s).strip()


def dataset_agrees(pred: str | None, gold: str | None) -> bool:
    if not pred or not gold:
        return False
    a, b = lookup(pred), lookup(canon_dataset(gold))
    if a is not None and b is not None and a.name == b.name:
        return True
    return E.names_agree(pred, canon_dataset(gold)) or E.names_agree(canon_dataset(pred), canon_dataset(gold))


def canon_metric(name: str | None) -> str:
    s = name or ""
    m = re.search(r"\(([A-Za-z0-9@%\- ]{2,8})\)", s)
    if m and m.group(1).replace(" ", "").isupper():
        return m.group(1)
    return re.sub(r"\([^)]*\)", " ", s).strip()


def metric_agrees(pred: str | None, gold: str | None, gold_dataset: str | None = None) -> bool | None:
    """None when the gold 'metric' is a split name that says nothing about the measure ("Matched", "Test", "Average").
    Two definitional mappings (a sixth revision of the scorer, 12 Oct, made when the repair began to write the benchmark's own measure into CoLA and STS-B profiles whose
    paper never names it): Papers with Code files CoLA's results under "Accuracy" although the numbers are Matthews correlations (GLUE defines CoLA by it), and a stored
    "correlation" that does not say which one agrees with a leaderboard's Pearson / Spearman correlation (it is less specific, not different)."""
    g = canon_metric(gold)
    if _norm(g) in UNSPECIFIED_METRICS or not _norm(g):
        return None
    p = canon_metric(pred)
    if bool(pred) and E.metrics_agree(p, g):
        return True
    if _norm(canon_dataset(gold_dataset)) == "cola" and _norm(g) in {"accuracy", "acc"} and "matthews" in _norm(p):
        return True
    if _norm(p) == "correlation" and re.search(r"pearson|spearman|correlation", g, re.I):
        return True
    return False


def _expand(name: str | None) -> str:
    s = re.sub(r"\b(ens|sgl)\b\.?", lambda m: ABBREVIATIONS[m.group(1).lower()], name or "", flags=re.I)
    return " ".join(ABBREVIATIONS.get(w.lower(), w) for w in re.findall(r"[A-Za-z0-9\-+\.]+", s))


def model_agrees(pred: str | None, gold: str | None) -> bool:
    if not pred or not gold:
        return False
    p, g = _expand(pred), _expand(gold)
    fp, fg = model_family(p), model_family(g)
    return bool(fp and fg and (fp == fg or fp in fg or fg in fp)) or E.names_agree(p, g)


def score_paper(gold: list[dict], profiles: list, numbers: set[str], text_compact: str = "") -> tuple[dict, list[dict], list[dict]]:
    rec = [g for g in gold if recoverable(g, numbers, text_compact)]
    cov = d = m = mo = ident = t = 0
    misses, bad = [], []
    for g in rec:
        printed = [float(f) for f in printed_forms(g) & numbers]
        cands = [p for p in profiles if p.value is not None
                 and (E.values_match(float(g["value"]), float(p.value)) or any(abs(f - float(p.value)) <= 0.0051 for f in printed))]
        if not cands:
            misses.append(g)
            continue
        cov += 1

        def fit(p):
            return (dataset_agrees(p.dataset, g["dataset"]), metric_agrees(p.metric, g["metric"], g["dataset"]) is not False, model_agrees(p.model, g["model"]),
                    E.tasks_agree(p.task, g["task"]))

        best = max(cands, key=lambda p: sum(fit(p)))
        fd, fm, fmo, ft = fit(best)
        d += fd
        m += fm
        mo += fmo
        t += ft
        ident += fd and fm and fmo
        if not (fd and fm and fmo):
            bad.append({"gold": g, "pred": {"dataset": best.dataset, "metric": best.metric, "task": best.task, "model": best.model, "setting": best.setting,
                                              "language": best.language, "profile_id": best.profile_id}, "agree": {"dataset": fd, "metric": fm, "model": fmo}})
    return {"gold": len(gold), "recoverable": len(rec), "covered": cov, "dataset": d, "metric": m, "model": mo, "identity": ident, "task": t}, bad, misses


def aggregate(per: dict[str, dict]) -> dict:
    tot = {k: sum(r[k] for r in per.values()) for k in ("gold", "recoverable", "covered", "dataset", "metric", "model", "identity", "task")}
    pct = lambda a, b: round(100 * a / b, 1) if b else 0.0
    return {**tot, "recoverable_pct": pct(tot["recoverable"], tot["gold"]), "coverage_pct": pct(tot["covered"], tot["recoverable"]),
            "dataset_pct": pct(tot["dataset"], tot["covered"]), "metric_pct": pct(tot["metric"], tot["covered"]), "model_pct": pct(tot["model"], tot["covered"]),
            "identity_pct": pct(tot["identity"], tot["covered"]), "task_pct": pct(tot["task"], tot["covered"]),
            "identity_end_to_end_pct": pct(tot["identity"], tot["recoverable"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--show", type=int, default=0, help="print this many disagreements and misses per paper (dev papers; the test papers only with --split test / all)")
    ap.add_argument("--paper", default=None, help="only this arXiv id")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = get_settings()
    index = load_chunk_index(cfg.paths.index_dir / "chunks.jsonl")
    by_title = load_rows()
    store = {False: ProfileStore(repaired=False), True: ProfileStore(repaired=True)}
    report: dict = {"split": args.split, "papers": {}, "variants": {}}
    per_variant = {"stored (raw extraction)": {}, "repaired (shipped)": {}}
    for pid, title in sorted(index.titles.items()):
        if (args.split != "all" and split_of(pid) != args.split) or (args.paper and pid != args.paper):
            continue
        gold, how = rows_of(title, by_title)
        if not gold:
            continue
        paper_text = "\n".join(t for c, t in index.texts.items() if c.split(":")[0] == pid)
        numbers, text_compact = printed_numbers(paper_text), _norm(paper_text)
        report["papers"][pid] = {"title": title, "split": split_of(pid), "match": how, "rows": len(gold)}
        for name, repaired in (("stored (raw extraction)", False), ("repaired (shipped)", True)):
            res, bad, misses = score_paper(gold, store[repaired].for_paper(pid), numbers, text_compact)
            per_variant[name][pid] = res
            if repaired and args.show:
                for b in bad[: args.show]:
                    g, p = b["gold"], b["pred"]
                    flags = "".join(k[0].upper() if not v else "." for k, v in b["agree"].items())
                    print(f"  [{flags}] {pid} {g['raw']:<8} gold: {g['task'][:30]} | {g['dataset'][:34]} | {g['metric'][:22]} | {g['model'][:26]}")
                    print(f"{'':36s}pred: {str(p['task'])[:30]} | {str(p['dataset'])[:34]} | {str(p['metric'])[:22]} | {str(p['model'])[:26]} | {p['setting']} | {p['language']}")
                for g in misses[: args.show]:
                    print(f"  [MISSING] {pid} {g['raw']:<8} gold: {g['task'][:30]} | {g['dataset'][:34]} | {g['metric'][:22]} | {g['model'][:26]}")
    print(f"papers matched: {len(report['papers'])} ({args.split}); rows: {sum(v['rows'] for v in report['papers'].values())}")
    for name, per in per_variant.items():
        a = aggregate(per)
        report["variants"][name] = {"aggregate": a, "per_paper": per}
        print(f"\n{name}: recoverable {a['recoverable']} of {a['gold']} rows ({a['recoverable_pct']} %) | coverage {a['coverage_pct']} % ({a['covered']} of {a['recoverable']}) | "
              f"agree on covered: data set {a['dataset_pct']}, metric {a['metric_pct']}, model {a['model_pct']}, identity (all three) {a['identity_pct']}, task {a['task_pct']} | "
              f"identity end to end {a['identity_end_to_end_pct']} %")
    print("\nper paper (repaired): rows, recoverable, covered, identity right")
    for pid, r in report["variants"]["repaired (shipped)"]["per_paper"].items():
        print(f"  {pid:24s} {report['papers'][pid]['split']:5s} {report['papers'][pid]['match']:9s} rows {r['gold']:3d} recoverable {r['recoverable']:3d} covered {r['covered']:3d} identity {r['identity']:3d}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
