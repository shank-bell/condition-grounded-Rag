# Backup Models on Hugging Face

As of 2026-09-27.

## Why these backups

These Hugging Face models can replace a pipeline component if the current one is too slow or too weak on the 6 GB laptop. Every repo below was checked against the [Hugging Face model API](https://huggingface.co/docs/hub/api) on 2026-09-27: it exists, is not gated, and its size and licence are as listed. None has been downloaded or tested, so quality is unverified. Model names live in `config.toml`, so most swaps are a one-line change.

## Backup models by component

Every pipeline component has at least one lighter or alternative model below; the extractor has the most options because it carries the most risk. Sizes are parameter counts from the API.

| Component | Tier | Model | Params | Licence |
| --- | --- | --- | --- | --- |
| Embedder (D, 4) | Current | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | ~568M (approx.) | MIT |
| Embedder (D, 4) | Lighter | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 109M | MIT |
| Embedder (D, 4) | Lighter | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 33M | MIT |
| Embedder (D, 4) | Lightest | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 23M | Apache-2.0 |
| Reranker (5) | Current | [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) | 23M | Apache-2.0 |
| Reranker (5) | Stronger | [cross-encoder/ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) | 33M | Apache-2.0 |
| Reranker (5) | Alternative | [mixedbread-ai/mxbai-rerank-xsmall-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) | 71M | Apache-2.0 |
| Reranker (5) | Stronger | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) | 278M | MIT |
| NLI (7, 9) | Current | [MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) | 184M | MIT |
| NLI (7, 9) | Lighter | [cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | 142M | Apache-2.0 |
| NLI (7, 9) | Lightest | [cross-encoder/nli-MiniLM2-L6-H768](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) | 82M | Apache-2.0 |
| NLI (7, 9) | Stronger | [MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) | 435M | MIT |
| Claim-check critic (9) | Purpose-built | [vectara/hallucination_evaluation_model](https://huggingface.co/vectara/hallucination_evaluation_model) | 110M | Apache-2.0 |
| Query classifier (1) | Current (needs fine-tuning) | [allenai/scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased) | ~110M (approx.) | not listed |
| Query classifier (1) | Zero-shot, no training | [MoritzLaurer/deberta-v3-base-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0) | 184M | MIT |
| Query classifier (1) | Zero-shot, heavier | [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) | 407M | MIT |
| Profile extractor (C) | Template extraction | [numind/NuExtract-1.5](https://huggingface.co/numind/NuExtract-1.5) | 3.8B | MIT |
| Profile extractor (C) | Template extraction, small | [numind/NuExtract-1.5-tiny](https://huggingface.co/numind/NuExtract-1.5-tiny) | not checked | not checked |
| Profile extractor (C) | Small LLM | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | 1.5B | Apache-2.0 |
| Profile extractor (C) | Small LLM | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | 3.1B | Qwen custom |
| Profile extractor (C) | Small LLM | [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.8B | MIT |
| Profile extractor (C) | No LLM (entities only) | [urchade/gliner_medium-v2.1](https://huggingface.co/urchade/gliner_medium-v2.1) | 195M | Apache-2.0 |
| Generator LLM (1, 3, 8) | Current (Ollama) | `gemma4:e4b` | 8.0B | Apache-2.0 |
| Generator LLM (1, 3, 8) | Lighter | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it); Ollama tag `gemma4:e2b` | not checked | not checked |

## Swap notes

Most swaps are a config change, but the embedder and the NLI models each need extra care.

- **Embedder:** a different model changes the vector size (1024 to 768 or 384), so rebuild the ChromaDB collection and re-embed every chunk. The BGE-en and MiniLM models are English-only, which is fine for English CS papers.
- **NLI models:** they order their labels differently. Read `model.config.id2label` instead of hardcoding indexes.
- **Claim-check critic:** the Vectara model scores whether a claim is supported by its context. It may need `trust_remote_code`, so check the model card and test it first.
- **Zero-shot classifier:** it replaces SciBERT fine-tuning, so no labelled data is needed. It is slower per query than a fine-tuned small classifier.
- **GLiNER:** it finds entity names (dataset, metric, model, language) but does not link numbers to them, so pair it with regex.
- **Licences:** Qwen2.5-3B-Instruct uses a custom licence. Read it before using the model in the report or paper.
- **Ollama:** GGUF repos on Hugging Face can be pulled with `ollama run hf.co/<user>/<repo>`. Use GGUF repos there, not raw weights. The `gemma4:e2b` tag exists in the [Ollama library](https://ollama.com/library/gemma4) (checked 2026-09-27).

## Profile extractor fallback order

The Condition Profile Extractor is the highest-risk component, because both contributions depend on its output. Move down this list if extraction is too slow, or if a hand-checked sample of about 10 papers shows poor field accuracy.

1. **`gemma4:e4b`**, the current model, with thinking off and temperature 0.
2. **`gemma4:e2b`**, same prompt and schema, expected to be faster.
3. **NuExtract-1.5 or NuExtract-1.5-tiny**, built to fill a JSON template from text. NuExtract-1.5 is 3.8B, so it will not be faster than `e4b`.
4. **Qwen2.5-3B or 1.5B Instruct**, run through Ollama with a JSON schema.
5. **GLiNER plus regex**, which extracts entity names and numbers without an LLM. Quality is lowest, but it runs on CPU in seconds.
