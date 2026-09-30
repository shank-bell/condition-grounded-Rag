"""Show what the extractor sends for one chunk and what the LLM answers: python scripts/debug_view.py 2302.13971:0014"""
import sys

from cgrag.ingestion.profile_extractor import (MAX_OUTPUT_TOKENS, NO_METHODS, SYSTEM, _SCHEMA, _grounded, _is_result, _normalize,
                                               _numbers, _views, salvage_profiles, link_methods_chunk)
from cgrag.ingestion.tables import is_structured_table
from cgrag.llm import OllamaLLM
from cgrag.stores.vector_store import VectorStore

sys.stdout.reconfigure(encoding="utf-8")
chunk_id = sys.argv[1]
store = VectorStore()
chunk = store.get([chunk_id])[0]
methods = link_methods_chunk(chunk, [c for c in store.paper_chunks(chunk.paper_id) if c.section == "methods" and not is_structured_table(c.text)])
llm = OllamaLLM()
numbers = _numbers(chunk.text)
for i, view in enumerate(_views(chunk.text), 1):
    prompt = f"METHODS CHUNK:\n{methods.text if methods else NO_METHODS}\n\nRESULTS CHUNK:\n{view}\n\nExtract all reported results."
    res = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], schema=_SCHEMA,
                   temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
    profiles, cut = salvage_profiles(res.text)
    print(f"##### view {i}: {len(view)} chars sent; reply {res.gen_tokens} tokens in {res.wall_s:.0f}s; parsed {len(profiles)} profiles (cut={cut})")
    print(view[:900])
    print("--- raw reply (first 700 chars):", res.text[:700].replace("\n", " "))
    for p in map(_normalize, profiles):
        why = "kept" if (_grounded(p, numbers) and _is_result(p)) else ("UNGROUNDED" if not _grounded(p, numbers) else "NOT-A-RESULT")
        print(f"   [{why}] model={p.model!r} dataset={p.dataset!r} metric={p.metric!r} value={p.value}")
