"""The data set of a table, read from the paper's own text (11 Oct 2026).

Why. The extractor sees one table at a time. Many papers say the data set once ("Extensive experiments were conducted on the Gigaword dataset") and then print
tables whose captions do not ("Table 2: Results of different interaction approaches"), or print one column per task ("Chunk F1", "NER F1") and leave the
benchmark of each task to the experiment section. The stored profile then has no data set, or a task name or a metric label in its place (on the public
benchmark of docs 4.1c these are 409 of the 784 data set names that disagree with the human gold). The repair rules cannot settle that from the table.

What. For every group of profiles whose data set is missing, a task name, a metric / split label or a cut-off fragment, one question goes to the local LLM
with the paper's own sentences about data sets, the sentences that mention the table and the table's caption and first rows: which data set were these
numbers measured on? The answer is kept only when the paper itself carries it (the name occurs in the paper's text, the LLM quotes the words that say it, it is
one name and not a list, and not a word that is a task, a metric or a language). Every decision is stored with its evidence sentence, so a reader can check
it, and the repair applies it as the rule `dataset_context` (ingestion/repair.py). A group the paper does not settle stays as it was.

The resolution is NOT part of the code-only repair: it calls a language model once per group (about 3 s), is written to `<index>/dataset_context.json` and
can be switched off with `[features] dataset_context`.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..knowledge.benchmarks import families, is_language, lookup, lookup_cut, task_fits
from ..pipeline.conditions import DATASET_SUITES, norm
from ..schemas import ConditionProfile

_METRIC_LABEL = re.compile(r"^(?:avg|average|mean|overall|all|f-?score|f1|f|acc|accuracy|bleu|rouge[\w-]*|em|las|uas|v-?m|p|r|score|test|dev|dev set|test set|"
                           r"fuzzy b-?cubed|b-?cubed|precision|recall)\.?$", re.I)
_GENERIC = re.compile(r"^(?:data ?sets?|corpus|corpora|benchmarks?|test sets?|training sets?|dev sets?|the data|unknown|none|n/?a|[a-z ]*task[a-z ]*)$", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_DATA_WORD = re.compile(r"data ?set|corpus|benchmark|treebank|evaluat|experiment|shared task|test set|we (?:use|train|test|conduct|perform)|results on", re.I)
_NAME_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9\-]{2,}\b")

SYSTEM = ("You read the text of a research paper and name the data set (benchmark) on which the numbers of ONE table or ONE task were measured. "
          "Answer with the name exactly as the paper writes it (for example 'Gigaword', 'CoNLL-2003', 'CoNLL03', 'WSJ'). "
          "Answer UNKNOWN when the table lists several data sets as columns, when the paper does not say, or when you are not sure. "
          "Never invent a name that the text does not contain, and never answer with a list.")


def label_kind(name: str | None) -> str | None:
    """Why a stored data set value cannot be trusted: 'none', 'metric_label', 'task_name' (a task or a column of tasks, not a benchmark), 'fragment'
    (a piece of a name cut at a column border); None when the value looks like a data set name."""
    if name is None or not str(name).strip():
        return "none"
    s = str(name).strip()
    if _METRIC_LABEL.match(s):
        return "metric_label"
    if lookup(s) is not None:
        return None
    if families(s) and len(s.split()) <= 3:
        return "task_name"
    if len(s) <= 3 or re.search(r"(?:^|\s)[a-z]{1,2}$", s) or (len(s) <= 5 and re.match(r"^[A-Za-z]{1,3}\b", s) and not s.isupper()):
        return "fragment"
    return None


def compact(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


@dataclass
class Resolution:
    dataset: str
    evidence: str
    mode: str
    kind: str


def group_key(chunk_id: str, label: str | None) -> str:
    return f"{chunk_id}|{(label or '').strip()}"


def groups(profiles: list[ConditionProfile]) -> dict[str, list[ConditionProfile]]:
    """Profiles whose data set cannot be trusted, by (chunk, stored value)."""
    out: dict[str, list[ConditionProfile]] = {}
    for p in profiles:
        if label_kind(p.dataset):
            out.setdefault(group_key(p.chunk_id, p.dataset), []).append(p)
    return out


def paper_sentences(chunks: dict[str, str]) -> list[str]:
    """The prose sentences of the paper (table rows left out)."""
    out: list[str] = []
    for text in chunks.values():
        if text.lstrip().startswith("Table") and text.count("|") > 20:
            continue
        for line in text.splitlines():
            if line.lstrip().startswith("|"):
                continue
            out += [s.strip() for s in _SENT.split(line) if len(s.strip()) > 30]
    return out


def table_number(caption: str) -> str | None:
    m = re.match(r"\s*Table\s+([0-9A-Za-z.]+)", caption or "")
    return m.group(1).rstrip(".:") if m else None


def build_prompt(title: str, sentences: list[str], chunk_text: str, label: str | None, kind: str, mode: str) -> str:
    lines = chunk_text.splitlines()
    caption = lines[0] if lines else ""
    rows = "\n".join(lines[:5])[:900]
    n = table_number(caption)
    refs = [s for s in sentences if n and re.search(rf"\bTable\s+{re.escape(n)}\b", s)][:3]
    words = [w for w in re.findall(r"[A-Za-z]{3,}", label or "") if w.lower() not in {"the", "and", "for"}]
    about = [s for s in sentences if words and any(re.search(rf"\b{re.escape(w)}", s, re.I) for w in words)][:3] if mode == "label" else []
    data = [s for s in sentences if _DATA_WORD.search(s) and _NAME_TOKEN.search(s)]
    data = sorted(data, key=lambda s: -(len(_NAME_TOKEN.findall(s)) + 3 * bool(re.search(r"data ?set|corpus|benchmark", s, re.I))))[:7]
    text = "\n".join(f"- {s[:330]}" for s in dict.fromkeys(refs + about + data))[:2800]
    if mode == "label":
        question = (f"In this paper, which data set (benchmark) is used for '{label}'? ('{label}' is how the extractor labelled the numbers; it may be a task, a "
                    "metric or a column heading and is not a data set name.)")
    else:
        question = "Which data set were the numbers of this table measured on?"
    return (f"PAPER TITLE: {title}\n\nSENTENCES FROM THE PAPER:\n{text}\n\nTABLE:\n{rows}\n\n{question} "
            "Reply as JSON: {\"dataset\": \"<name or UNKNOWN>\", \"evidence\": \"<the words of the paper that say it>\"}")


_TRAILING = re.compile(r"\s+(?:ner|pos|chunking|parsing|tagging|dataset|data set|corpus|task|benchmark|test set|dev set|treebank data)$", re.I)


def clean_name(name: str) -> str:
    """The data set without the task word the paper writes after it ("the CoNLL03 NER dataset" -> "CoNLL03"); "Penn Treebank" keeps its own word."""
    s = re.sub(r"^(?:the|a|an)\s+", "", (name or "").strip().strip("\"'"), flags=re.I)
    for _ in range(2):
        shorter = _TRAILING.sub("", s).strip()
        if len(compact(shorter)) < 3 or shorter == s:
            break
        s = shorter
    return s


def accept(name: str, evidence: str, paper_text_compact: str, kind: str, label: str | None = None, shown: str = "", tasks: list[str] | None = None) -> bool:
    """The paper must carry the answer: one name, grounded in the text and in what the model was shown or quoted, not a task / metric / language word,
    and not a replacement that would lose what the stored label says (found on 28 papers: "MASSIVE (Intent)" -> "MASSIVE", "StoryCloze 2016" ->
    "StoryCloze", "CB" -> "SuperGLUE", "MRPC STS" -> "MRPC")."""
    s = (name or "").strip()
    c = compact(s)
    if not s or s.upper().startswith("UNKNOWN") or len(c) < 3 or len(s) > 48:
        return False
    if _GENERIC.match(s) or is_language(s) or _METRIC_LABEL.match(s):
        return False
    if re.search(r"\band\b|&|,|;|\bor\b", s, re.I) and lookup(s) is None:
        return False
    if families(s) and lookup(s) is None and len(s.split()) <= 2 and not re.search(r"\d", s):
        return False
    if c not in paper_text_compact:
        return False
    if label and c in compact(label):
        return False
    if kind in ("task_name", "fragment"):
        if norm(s) in DATASET_SUITES:
            return False
        if label and any(lookup(w) is not None for w in re.findall(r"[A-Za-z0-9][\w\-.]*", label) if len(w) > 2):
            return False
    if kind == "fragment" and label:
        piece = compact(clean_name(label))
        if piece and not (c.startswith(piece) or c.endswith(piece)):
            return False
    bench = lookup(s)
    if bench is not None and tasks:
        fits = [task_fits(x, bench) for x in tasks if x]
        if fits and all(f is False for f in fits):
            return False
    return c in compact(evidence) or c in compact(shown)


def resolve_paper(llm, profiles: list[ConditionProfile], chunks: dict[str, str], title: str = "", progress=None) -> dict[str, Resolution]:
    """Resolutions for the untrustworthy data set values of one paper. `profiles` are the repaired profiles of the paper."""
    todo = groups(profiles)
    if not todo:
        return {}
    sentences = paper_sentences(chunks)
    paper_text = compact(" ".join(chunks.values()))
    schema = {"type": "object", "properties": {"dataset": {"type": "string"}, "evidence": {"type": "string"}}, "required": ["dataset", "evidence"]}
    out: dict[str, Resolution] = {}
    for key, members in todo.items():
        chunk_id, label = key.split("|", 1)
        kind = label_kind(label) or "none"
        mode = "label" if kind in ("task_name", "metric_label") else "table"
        prompt = build_prompt(title, sentences, chunks.get(chunk_id, ""), label or None, kind, mode)
        try:
            reply = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], schema=schema, temperature=0.0, max_tokens=140)
            data = json.loads(reply.text)
        except Exception:
            continue
        name, evidence = clean_name(data.get("dataset") or ""), (data.get("evidence") or "").strip()
        if name and lookup(name) is None and lookup_cut(name) is not None:
            name = lookup_cut(name).name
        if progress:
            progress(key, name, evidence)
        if accept(name, evidence, paper_text, kind, label or None, prompt, [m.task for m in members if m.task]):
            out[key] = Resolution(dataset=name, evidence=evidence[:240], mode=mode, kind=kind)
    return out


def load_resolutions(path: Path) -> dict[str, dict[str, dict]]:
    """{paper_id: {group key: {dataset, evidence, mode, kind}}}; empty when the file does not exist."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_resolutions(path: Path, data: dict[str, dict[str, dict]]) -> None:
    Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


__all__ = ["Resolution", "accept", "clean_name", "groups", "label_kind", "load_resolutions", "resolve_paper", "save_resolutions", "asdict"]
