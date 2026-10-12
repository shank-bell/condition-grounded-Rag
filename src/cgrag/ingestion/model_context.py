"""The system behind a row that a table labels with a citation (11 Oct 2026).

Papers print the systems they compare as a name ("mBERT") or as the work that introduced them ("Devlin et al. (2018)", "Lample and Conneau (2019)"). The
extractor stores the label as the model, so the same system is "mBERT" in one paper and "Devlin et al.(2018)" in another and the two are never compared
(the XLM-R paper's mBERT rows were not recognised as mBERT; 253 stored profiles of the 28 papers carry such a label).

Two steps, both grounded in the paper's own words:

* `strip_citation` (code only): "Adaptive (Sukhbaatar et al., 2019)" is the system "Adaptive"; the citation after a name is not part of the name.
* `resolve_models` (one LLM question per label and paper): "Devlin et al. (2018)" is the system the paper writes next to that citation ("multilingual
  BERT (mBERT; Devlin et al., 2018)"). The answer is kept only when the name occurs in the paper's text, is not a surname or a word of the citation, and is
  quoted in the evidence or in the sentences the model was shown.

Decisions are stored with the data set decisions (ingestion/dataset_context.py) under the key `model|<label>`.
"""
from __future__ import annotations

import json
import re

from ..schemas import ConditionProfile
from .dataset_context import compact

_YEAR = r"(?:19|20)\d\d[a-z]?"
_SURNAME = r"[A-Z][a-zÀ-ſ][A-Za-zÀ-ſ\-']*"
_ET_AL = r"(?:et\.(?:\s?(?:al|l)\.?)?|et\s?al\.?)"
PURE_CITATION = re.compile(rf"^\s*{_SURNAME}(?:\s+(?:and|&)\s+{_SURNAME}|\s+{_ET_AL})?[\s,.]*[(\[]?\s*{_YEAR}\s*[)\]]?\s*$")
NAME_CITATION = re.compile(rf"^(?P<name>.+?)\s*[\[(]\s*[^()\[\]]*?(?:{_ET_AL}|{_YEAR})[^()\[\]]*[\])]\s*$")
_GENERIC = re.compile(r"^(?:model|models|baseline|system|method|approach|unknown|none|the model|ours?|sota|nmt|smt|mt|nlp|ml|ai|lm|mlm|clm|tlm|bpe|state[- ]of[- ]the[- ]art|previous work|prior work)$", re.I)
_CANONICAL = {"multilingualbert": "mBERT"}


def model_key(label: str) -> str:
    return f"model|{label.strip()}"


def is_pure_citation(model: str | None) -> bool:
    return bool(model) and PURE_CITATION.match(model) is not None


def strip_citation(model: str | None) -> str | None:
    """"Adaptive (Sukhbaatar et al., 2019)" -> "Adaptive"; None when the label is not a name followed by a citation (or is a bare citation)."""
    if not model or is_pure_citation(model):
        return None
    m = NAME_CITATION.match(model)
    if not m:
        return None
    name = m.group("name").strip(" ,;:-")
    if len(compact(name)) < 2 or re.search(_ET_AL + r"$", name, re.I) or PURE_CITATION.match(name) or _GENERIC.match(name):
        return None
    if re.search(_YEAR, name) or re.search(_ET_AL, name, re.I):
        return None
    return name


def citation_labels(profiles: list[ConditionProfile]) -> dict[str, list[ConditionProfile]]:
    out: dict[str, list[ConditionProfile]] = {}
    for p in profiles:
        if is_pure_citation(p.model):
            out.setdefault(p.model.strip(), []).append(p)
    return out


SYSTEM = ("You read the text of a research paper. Some rows of its tables are labelled with a citation instead of a name. Say which system or model the paper "
          "means by that citation, using the name the paper itself uses for it (for example 'mBERT', 'XLM', 'Unicoder'). Answer UNKNOWN when the text does not "
          "say or you are not sure. Never invent a name that the text does not contain.")


def _sentences_about(sentences: list[str], label: str) -> list[str]:
    surname = re.match(_SURNAME, label.strip())
    year = re.search(_YEAR, label)
    out = []
    for s in sentences:
        if surname and surname.group(0) in s and (not year or year.group(0)[:4] in s):
            out.append(s)
    for s in sentences:
        if surname and surname.group(0) in s and s not in out:
            out.append(s)
    return out[:6]


def build_prompt(title: str, sentences: list[str], label: str) -> str:
    about = "\n".join(f"- {s[:330]}" for s in _sentences_about(sentences, label))[:2200] or "(no sentence names this citation)"
    return (f"PAPER TITLE: {title}\n\nSENTENCES THAT MENTION THE CITATION:\n{about}\n\nA table row is labelled '{label}'. Which system or model does the paper mean? "
            "Reply as JSON: {\"model\": \"<name or UNKNOWN>\", \"evidence\": \"<the words of the paper that say it>\"}")


def accept(name: str, evidence: str, paper_text_compact: str, label: str, shown: str = "") -> bool:
    s = (name or "").strip()
    c = compact(s)
    if not s or s.upper().startswith("UNKNOWN") or len(c) < 2 or len(s) > 40 or _GENERIC.match(s):
        return False
    if re.search(r"\band\b|&|,|;", s, re.I) or PURE_CITATION.match(s) or re.search(_ET_AL, s):
        return False
    surname = re.match(_SURNAME, label.strip())
    if surname and compact(surname.group(0)) in c and len(c) <= len(compact(surname.group(0))) + 2:
        return False
    if c not in paper_text_compact:
        return False
    return c in compact(evidence) or c in compact(shown)


def resolve_models(llm, profiles: list[ConditionProfile], sentences: list[str], paper_text_compact: str, title: str = "",
                   progress=None) -> dict[str, dict]:
    """Decisions for the citation labels of one paper (`sentences`: its prose sentences; `paper_text_compact`: its whole text, compacted)."""
    out: dict[str, dict] = {}
    schema = {"type": "object", "properties": {"model": {"type": "string"}, "evidence": {"type": "string"}}, "required": ["model", "evidence"]}
    for label in citation_labels(profiles):
        prompt = build_prompt(title, sentences, label)
        try:
            reply = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], schema=schema, temperature=0.0, max_tokens=120)
            data = json.loads(reply.text)
        except Exception:
            continue
        name, evidence = (data.get("model") or "").strip(), (data.get("evidence") or "").strip()
        if progress:
            progress(model_key(label), name, evidence)
        if accept(name, evidence, paper_text_compact, label, prompt):
            name = _CANONICAL.get(compact(name), name)
            out[model_key(label)] = {"model": name, "evidence": evidence[:240], "mode": "model", "kind": "citation"}
    return out
