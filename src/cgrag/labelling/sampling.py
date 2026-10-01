"""Which rows go into the labelling sheets.

  job A  profiles to check (random, a few per table so every table is seen) and the source row each one was read from
  job C  pairs of results from two papers (same model family, dataset, metric; numbers > 2 % apart), picked so that every kind of
         difference is in the sample - the system's own verdict is only used to spread the sample, never shown to the labellers
  job B  the corpus map: what the extractor recorded per paper / dataset / language / model, so a question writer can look up
         whether a condition is covered (a lookup aid built from the extractor's output, NOT a source of truth)
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from ..pipeline.conditions import _LANG_ALIASES, metric_key, model_family, norm, observed_values, split_version
from ..pipeline.contradiction import classify, rel_diff, same_subject
from ..schemas import ConditionProfile

EASY_PAPERS = {"1810.04805": "BERT", "1806.03822": "SQuAD 2.0", "1809.05053": "XNLI", "1911.02116": "XLM-R", "1804.07461": "GLUE",
               "2103.10730": "MuRIL"}
HARD_PAPERS = {"1910.10683": "T5", "2005.14165": "GPT-3", "2302.13971": "LLaMA", "2307.09288": "Llama 2"}
ARXIV_PDF = "https://arxiv.org/pdf/{paper}#page={page}"


def arxiv_link(paper_id: str, page: int) -> str:
    return ARXIV_PDF.format(paper=paper_id, page=max(int(page), 1))


# ---------- source texts ----------

@dataclass
class ChunkIndex:
    texts: dict[str, str]
    pages: dict[str, int]
    titles: dict[str, str]          # paper id -> title as the system shows it


def export_chunks(path: Path) -> None:
    """Write every chunk (id, paper, title, page, section, text) of the index to a JSON-lines file, so that scripts that only read
    passages never open ChromaDB (it must not be opened by two processes that write)."""
    from ..stores.vector_store import VectorStore
    rows = [{"chunk_id": c.chunk_id, "paper_id": c.paper_id, "paper_title": c.paper_title, "page": c.page, "section": c.section,
             "text": c.text} for c in VectorStore().all_chunks()]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def load_chunk_index(path: Path) -> ChunkIndex:
    """Texts, pages and titles of all chunks from the export (re-exported from ChromaDB when it is missing or has no page numbers)."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    if not rows or "page" not in rows[0]:
        export_chunks(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    titles: dict[str, str] = {}
    for r in rows:
        if r.get("paper_title"):
            titles.setdefault(r["paper_id"], " ".join(r["paper_title"].split()))
    return ChunkIndex({r["chunk_id"]: r["text"] for r in rows}, {r["chunk_id"]: int(r["page"]) for r in rows}, titles)


def paper_titles(papers_dir: Path, cache: Path) -> dict[str, str]:
    """arXiv id -> title (the largest text on page 1), cached."""
    if cache.exists():
        found = json.loads(cache.read_text(encoding="utf-8"))
        if found:
            return found
    import fitz

    from ..ingestion.pdf_loader import _title
    out: dict[str, str] = {}
    for pdf in sorted(papers_dir.glob("*.pdf")):
        with fitz.open(pdf) as doc:
            out[pdf.stem] = " ".join(_title(doc, pdf.stem).split())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def value_forms(value: float) -> list[str]:
    forms = {f"{value:g}", f"{value:.1f}", f"{value:.2f}"}
    if float(value).is_integer():
        forms.add(f"{value:.0f}")
    return sorted(forms, key=len, reverse=True)


def _value_pattern(value: float) -> re.Pattern:
    return re.compile(r"(?<![\d.])(?:" + "|".join(re.escape(v) for v in value_forms(value)) + r")(?!\d)(?!\.\d)")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _compact(line: str, keep: list[int]) -> str:
    cs = _cells(line)
    out, last = [], -1
    for i in keep:
        if i >= len(cs) or i <= last:
            continue
        if i != last + 1:
            out.append("…")
        out.append(cs[i])
        last = i
    if last < len(cs) - 1:
        out.append("…")
    return "| " + " | ".join(out) + " |"


def evidence_excerpt(text: str, p: ConditionProfile, max_line: int = 230) -> str:
    """What a labeller needs to check a profile: the table caption, its header row(s) and the row the number was read from (the
    number is marked [[like this]]); for a prose chunk, the passage around the number."""
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    pat = _value_pattern(p.value)
    pipe = [ln for ln in lines if ln.lstrip().startswith("|")]
    if pipe:
        caption = next((ln for ln in lines if not ln.lstrip().startswith("|")), "")
        hits = [ln for ln in pipe if pat.search(ln)]
        want = norm(p.model)[:8]
        pick = next((ln for ln in hits if want and want in norm((_cells(ln) or [""])[0])), hits[0] if hits else None)
        header = [pipe[0]] + ([pipe[1]] if len(pipe) > 1 and pipe[1] is not pick and not re.search(r"\d+\.\d", pipe[1]) else [])
        if pick is None:
            return "\n".join([caption[:max_line], *[h[:max_line] for h in header], "(the number was not found in this table's text)"]).strip()
        pick_cells = _cells(pick)
        j = next((i for i, c in enumerate(pick_cells) if pat.search(c)), 0)
        if max(len(pick), *(len(h) for h in header)) > max_line:               # a wide table: first column + the value's column
            keep = sorted({0, max(j - 1, 0), j, j + 1})
            pick_cells[j] = pat.sub(lambda m: f"[[{m.group(0)}]]", pick_cells[j], count=1)
            marked = _compact("|" + "|".join(pick_cells) + "|", keep)
            header = [_compact(h, keep) for h in header]
        else:
            marked = pat.sub(lambda m: f"[[{m.group(0)}]]", pick, count=1)
        return "\n".join(x for x in [caption[:max_line], *header, marked] if x)
    flat = " ".join(text.split())
    m = pat.search(flat)
    if not m:
        return "(the number was not found in this passage)\n" + flat[:200]
    lo, hi = max(0, m.start() - 220), min(len(flat), m.end() + 120)
    return ("…" if lo else "") + flat[lo:m.start()] + f"[[{m.group(0)}]]" + flat[m.end():hi] + ("…" if hi < len(flat) else "")


# ---------- job A: profiles ----------

def sample_profiles(profiles: list[ConditionProfile], papers: list[str], per_paper: int, rnd: random.Random,
                    exclude: frozenset[str] | set[str] = frozenset(), cap_per_chunk: int = 6) -> list[ConditionProfile]:
    """per_paper random profiles of each paper, at most cap_per_chunk from one chunk so that many different tables are seen."""
    by_paper: dict[str, list[ConditionProfile]] = defaultdict(list)
    for p in profiles:
        if p.paper_id in papers and p.profile_id not in exclude and p.value is not None:
            by_paper[p.paper_id].append(p)
    out: list[ConditionProfile] = []
    for paper in papers:
        pool = sorted(by_paper[paper], key=lambda p: p.profile_id)
        rnd.shuffle(pool)
        taken, per_chunk = [], Counter()
        for p in pool:
            if len(taken) >= per_paper:
                break
            if per_chunk[p.chunk_id] < cap_per_chunk:
                taken.append(p)
                per_chunk[p.chunk_id] += 1
        for p in pool:
            if len(taken) >= per_paper:
                break
            if p not in taken:
                taken.append(p)
        out.extend(taken)
    return out


# ---------- job C: result pairs ----------

@dataclass
class Pair:
    x: ConditionProfile
    y: ConditionProfile
    verdict: str                  # the system's own classification (Stage 7's classify); kept in a private key file, never shown
    differing: list[str]
    reason: str
    rel: float
    key: tuple = field(default_factory=tuple)

    @property
    def stratum(self) -> str:
        if self.verdict == "GENUINE":
            return "genuine"
        if self.verdict == "NOT_COMPARABLE":
            return "not_comparable"
        d = set(self.differing)
        if "language" in d:
            return "explained_language"
        if "dataset_version" in d:
            return "explained_version"
        if d == {"setting"}:
            return "explained_setting"
        if "model_size" in d:
            return "explained_size"
        return "explained_other"


# how the sample is spread (sums to 50; scaled for another size). GENUINE is rare in this corpus, so all of them are taken.
QUOTA = {"genuine": 7, "not_comparable": 14, "explained_setting": 8, "explained_size": 8, "explained_version": 8, "explained_language": 5}


def mine_pairs(profiles: list[ConditionProfile], min_rel: float = 0.02) -> list[Pair]:
    """One pair per (pair of papers, subject): the two results of the same model family, dataset and metric whose values differ most."""
    groups: dict[tuple, list[ConditionProfile]] = defaultdict(list)
    for p in profiles:
        if p.value is None or p.value <= 0 or not p.dataset or not p.model:
            continue
        family, dataset, mk = model_family(p.model), norm(split_version(p.dataset)[0]), metric_key(p.metric)
        if len(family) >= 3 and len(dataset) >= 3 and mk:
            groups[(family, dataset, mk)].append(p)
    pairs: list[Pair] = []
    for key, members in sorted(groups.items()):
        by_paper: dict[str, list[ConditionProfile]] = defaultdict(list)
        for p in members:
            by_paper[p.paper_id].append(p)
        for a, b in combinations(sorted(by_paper), 2):
            best = None
            for x in by_paper[a]:
                for y in by_paper[b]:
                    if same_subject(x, y) and rel_diff(x.value, y.value) > min_rel:
                        d = rel_diff(x.value, y.value)
                        if best is None or d > best[0]:
                            best = (d, x, y)
            if best:
                d, x, y = best
                verdict, differing, reason = classify(x, y)
                pairs.append(Pair(x, y, verdict, differing, reason, d, key))
    return pairs


def pick_pairs(pairs: list[Pair], n: int, rnd: random.Random) -> list[Pair]:
    """n pairs spread over the strata of QUOTA, with caps so that no model, dataset or pair of papers dominates (relaxed if needed)."""
    scale = n / sum(QUOTA.values())
    quota = {k: max(1, round(v * scale)) for k, v in QUOTA.items()}
    by_stratum: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        by_stratum[p.stratum].append(p)
    for items in by_stratum.values():
        items.sort(key=lambda p: (p.key, p.x.paper_id, p.y.paper_id))
        rnd.shuffle(items)
    chosen: list[Pair] = []

    def caps_ok(p: Pair, slack: int) -> bool:
        family, dataset = p.key[0], p.key[1]
        papers = frozenset((p.x.paper_id, p.y.paper_id))
        return (sum(1 for c in chosen if c.key[0] == family) < 4 + slack and sum(1 for c in chosen if c.key[1] == dataset) < 6 + slack
                and sum(1 for c in chosen if frozenset((c.x.paper_id, c.y.paper_id)) == papers) < 3 + slack)

    for slack in (0, 2, 6, 100):
        for stratum, want in quota.items():
            have = sum(1 for c in chosen if c.stratum == stratum)
            for p in by_stratum.get(stratum, []):
                if have >= want:
                    break
                if p not in chosen and caps_ok(p, slack):
                    chosen.append(p)
                    have += 1
    leftovers = [p for p in pairs if p not in chosen]
    rnd.shuffle(leftovers)
    for slack in (2, 6, 100):
        for p in leftovers:
            if len(chosen) >= n:
                break
            if p not in chosen and caps_ok(p, slack):
                chosen.append(p)
    return chosen[:n]


# ---------- job B: the corpus map ----------

def language_label(raw: str | None) -> str | None:
    """'kn' -> 'Kannada', 'hindi' -> 'Hindi'; None for junk such as '7 languages' or 'multi-language'."""
    if not raw:
        return None
    r = _LANG_ALIASES.get(raw.strip().lower(), raw.strip())
    return r.capitalize() if re.fullmatch(r"[A-Za-z]{3,20}", r) and r.lower() not in {"language", "languages", "multilingual", "other", "indian"} else None


def _top(counter: Counter, n: int) -> str:
    return ", ".join(v for v, _ in counter.most_common(n))


def corpus_map(profiles: list[ConditionProfile], titles: dict[str, str]) -> dict[str, tuple[list[str], list[list]]]:
    """Lookup tables for the question writers: sheet name -> (headers, rows). A cell ("text", url) is written as a hyperlink."""
    papers: dict[str, list[ConditionProfile]] = defaultdict(list)
    datasets: dict[str, list[ConditionProfile]] = defaultdict(list)
    models: dict[str, list[ConditionProfile]] = defaultdict(list)
    spelled: dict[str, Counter] = defaultdict(Counter)
    spelled_model: dict[str, Counter] = defaultdict(Counter)
    for p in profiles:
        papers[p.paper_id].append(p)
        if p.dataset and 3 <= len(p.dataset) <= 40:
            name = split_version(p.dataset)[0]
            key = norm(name)
            if len(key) >= 3:
                datasets[key].append(p)
                spelled[key][name] += 1
        if p.model and 3 <= len(p.model) <= 40:
            fam = model_family(p.model)
            if len(fam) >= 3:
                models[fam].append(p)
                spelled_model[fam][p.model] += 1

    def langs(ps: list[ConditionProfile]) -> Counter:
        return Counter(l for p in ps if (l := language_label(p.language)))

    paper_rows = []
    for pid in sorted(papers):
        ps = papers[pid]
        paper_rows.append([pid, titles.get(pid, ""), len(ps), _top(Counter(p.task for p in ps if p.task), 5),
                           _top(Counter(split_version(p.dataset)[0] for p in ps if p.dataset), 8), _top(langs(ps), 10),
                           _top(Counter(p.model for p in ps if p.model), 8), ("open", arxiv_link(pid, 1))])
    sheets = {"Papers": (["arXiv id", "title", "profiles", "tasks", "datasets", "languages", "models", "paper"], paper_rows)}

    dataset_rows = []
    for key, ps in sorted(datasets.items(), key=lambda kv: -len(kv[1])):
        if len(ps) < 3:
            continue
        dataset_rows.append([spelled[key].most_common(1)[0][0], len(ps), len({p.paper_id for p in ps}),
                             ", ".join(sorted({p.paper_id for p in ps})), ", ".join(sorted(langs(ps))),
                             _top(Counter(p.model for p in ps if p.model), 12),
                             ", ".join(sorted({p.dataset_version for p in ps if p.dataset_version}))[:80],
                             _top(Counter(p.setting for p in ps if p.setting), 5), _top(Counter(p.metric for p in ps if p.metric), 5)])
    sheets["Datasets"] = (["dataset", "profiles", "papers", "paper ids", "languages recorded", "models recorded (top 12)",
                           "versions recorded", "settings recorded (top 5)", "metrics (top 5)"], dataset_rows)

    # dataset x language (counts of recorded results): a blank cell = nothing recorded for that language on that dataset
    language_total = Counter(l for ps in datasets.values() for l in langs(ps).elements())
    columns = [l for l, _ in language_total.most_common(45)]
    matrix_rows = []
    for key, ps in sorted(datasets.items(), key=lambda kv: -len(kv[1])):
        lc = langs(ps)
        if len(ps) >= 15 and len(lc) >= 2:
            matrix_rows.append([spelled[key].most_common(1)[0][0], len(ps), *[lc.get(l) or None for l in columns]])
    sheets["Dataset x Language"] = (["dataset", "profiles", *columns], matrix_rows)

    model_rows = []
    for fam, ps in sorted(models.items(), key=lambda kv: -len(kv[1])):
        if len(ps) < 3:
            continue
        model_rows.append([spelled_model[fam].most_common(1)[0][0], len(ps), len({p.paper_id for p in ps}),
                           ", ".join(sorted({p.paper_id for p in ps})),
                           _top(Counter(split_version(p.dataset)[0] for p in ps if p.dataset), 10),
                           ", ".join(sorted({p.model_size for p in ps if p.model_size and len(p.model_size) <= 12}))[:80]])
    sheets["Models"] = (["model (family)", "profiles", "papers", "paper ids", "datasets (top 10)", "sizes recorded"], model_rows)

    top_models = [fam for fam, ps in sorted(models.items(), key=lambda kv: -len(kv[1]))][:30]
    top_datasets = [key for key, ps in sorted(datasets.items(), key=lambda kv: -len(kv[1]))][:30]
    mm_rows = []
    for fam in top_models:
        counts = Counter(norm(split_version(p.dataset)[0]) for p in models[fam] if p.dataset)
        mm_rows.append([spelled_model[fam].most_common(1)[0][0], *[counts.get(k) or None for k in top_datasets]])
    sheets["Model x Dataset"] = (["model", *[spelled[k].most_common(1)[0][0] for k in top_datasets]], mm_rows)
    return sheets
