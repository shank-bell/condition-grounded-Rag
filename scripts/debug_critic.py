"""Why did the critic reject a sentence? Runs a question and, for every unsupported claim, lists the recorded results that carry
the claim's numbers and which test of Stage 9's profile match each one fails.

  python scripts/debug_critic.py "How well do models perform on Kannada NLI?"
"""
from __future__ import annotations

import argparse
import re

from cgrag.pipeline.conditions import metric_key, model_family, norm, setting_tags
from cgrag.pipeline.critic import _NUMBER
from cgrag.pipeline.run import Pipeline
from cgrag.pipeline.text import plain


def why_not(claim: str, p) -> str:
    numbers, words, flat, tags = set(_NUMBER.findall(claim)), set(re.findall(r"[a-z0-9]+", claim.lower())), norm(claim), setting_tags(claim)
    if f"{p.value:g}" not in numbers:
        return "number differs"
    if not p.metric:
        return "no metric recorded"
    metric = norm(p.metric)
    if not ((len(metric) >= 3 and metric in flat) or metric_key(p.metric) in words or metric in words):
        return f"metric '{p.metric}' is not named in the claim"
    family = model_family(p.model)
    if p.model and len(family) >= 3 and family not in flat:
        return f"model '{p.model}' (family '{family}') is not named in the claim"
    recorded = setting_tags(p.setting)
    if tags and recorded and not (tags & recorded):
        return f"setting: claim {sorted(tags)} vs recorded {sorted(recorded)}"
    return "MATCHES"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    args = ap.parse_args()
    pipe = Pipeline()
    r = pipe.run(args.question)
    ids = [s.chunk_id for s in r.sources]
    profiles = pipe.profiles.for_chunks(ids)
    bad = [c for c in r.claim_checks if not c.supported]
    print(f"{len(r.claim_checks)} claims, {len(bad)} unsupported; sources {ids}")
    for c in bad:
        print(f"\nUNSUPPORTED: {c.sentence}")
        numbers = set(_NUMBER.findall(plain(c.sentence)))
        found = False
        for cid, plist in profiles.items():
            for p in plist:
                if f"{p.value:g}" in numbers:
                    found = True
                    print(f"   {cid}: model={p.model!r} metric={p.metric!r} value={p.value:g} setting={p.setting!r} lang={p.language!r} "
                          f"dataset={p.dataset!r} -> {why_not(plain(c.sentence), p)}")
        if not found:
            print("   no recorded result with these numbers in the cited sources")


if __name__ == "__main__":
    main()
