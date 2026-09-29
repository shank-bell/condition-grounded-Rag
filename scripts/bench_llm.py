"""Benchmark the local LLM on a profile-extraction prompt (synthetic test text, not a real paper)."""
from __future__ import annotations

import subprocess
import sys

from cgrag.config import get_settings
from cgrag.llm import OllamaLLM
from cgrag.schemas import ProfileExtraction

SYSTEM = (
    "You extract experimental results from computer-science research papers. "
    "For every distinct reported result (one metric value), output one profile. "
    "Use only facts stated in the text. If a field is not stated, use null - never guess. "
    "'value' is the number only (no % sign). 'model_size' is the parameter count or size label "
    "(e.g. 110M, base, large). 'language' is the language the model was evaluated on. "
    "'setting' is the evaluation regime (e.g. zero-shot, fine-tuned, few-shot). "
    "'evidence' is a short verbatim quote containing the number."
)

RESULTS = (
    "Table 3 reports F1 on the dev sets. On SQuAD v1.1, BERT-large reaches 90.9 F1 and BERT-base "
    "reaches 88.5 F1. On SQuAD v2.0, BERT-large reaches 81.8 F1 while BERT-base reaches 76.3 F1. "
    "On the multilingual XNLI benchmark, mBERT fine-tuned on English training data obtains 74.3% "
    "accuracy on Hindi and 68.1% accuracy on Swahili in the zero-shot cross-lingual transfer setting."
)
METHODS = (
    "We fine-tune BERT-base (110M parameters) and BERT-large (340M parameters) for 3 epochs with a "
    "learning rate of 3e-5 and batch size 32. mBERT is the multilingual BERT-base checkpoint."
)


def main(n_runs: int = 3) -> None:
    cfg = get_settings()
    llm = OllamaLLM()
    prompt = f"METHODS CHUNK:\n{METHODS}\n\nRESULTS CHUNK:\n{RESULTS}\n\nExtract all reported results."
    print(f"model={cfg.llm.model} think={cfg.llm.think} num_ctx={cfg.llm.num_ctx}\n")
    for i in range(n_runs):
        out, res = llm.structured(prompt, ProfileExtraction, system=SYSTEM)
        label = "(includes model load)" if i == 0 else ""
        print(
            f"run {i + 1}: {res.wall_s:5.1f}s  prompt={res.prompt_tokens} tok  "
            f"gen={res.gen_tokens} tok  {res.tokens_per_s:5.1f} tok/s {label}"
        )
        if i == n_runs - 1:
            print(f"\nextracted {len(out.profiles)} profiles:")
            for p in out.profiles:
                print(" ", p.model_dump(exclude_none=True, exclude={"evidence"}))
    print("\n--- ollama ps ---")
    print(subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
