from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]


class LLMConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "gemma4:e4b"
    think: bool = False
    num_ctx: int = 8192
    keep_alive: str = "15m"
    extract_temperature: float = 0.0
    generate_temperature: float = 0.2
    parallel: int = 1            # concurrent requests during offline extraction (the Ollama server must allow it)
    request_timeout: float = 240.0   # seconds; a wedged Ollama runner raises instead of blocking the pipeline forever


class ModelsConfig(BaseModel):
    embedder: str = "BAAI/bge-m3"
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    query_classifier: str = "allenai/scibert_scivocab_uncased"


class DevicesConfig(BaseModel):
    embedder: str = "cpu"
    reranker: str = "cpu"
    nli: str = "cpu"
    classifier: str = "cpu"


class PathsConfig(BaseModel):
    papers_dir: Path = Path("data/papers")
    index_dir: Path = Path("data/index")
    profile_db: Path = Path("data/index/profiles.sqlite")
    chroma_dir: Path = Path("data/index/chroma")
    bm25_path: Path = Path("data/index/bm25.pkl")


class ChunkingConfig(BaseModel):
    chunk_size: int = 2000
    chunk_overlap: int = 200


class RetrievalConfig(BaseModel):
    dense_k: int = 30
    bm25_k: int = 30
    rrf_k: int = 60
    fused_top: int = 20
    rerank_keep: int = 10
    rerank_threshold: float = -2.0
    section_boost: float = 0.15
    generate_top: int = 5
    use_cards: bool = True       # retrieval cards (ingestion/cards.py): a card vector as a third ranking + keyword index + reranker text


class ApplicabilityConfig(BaseModel):
    max_reretrieve: int = 1


class ContradictionConfig(BaseModel):
    nli_threshold: float = 0.70
    numeric_rel_diff: float = 0.02
    max_pairs: int = 45


class CriticConfig(BaseModel):
    entail_threshold: float = 0.50
    max_regenerate: int = 1


class AgentsConfig(BaseModel):
    """A different open-weights model per LLM use case (any Ollama tag, e.g. hf.co/<user>/<repo>). Empty = [llm].model,
    which is also the model that writes the answer (Stage 8)."""
    orchestrator_model: str = ""      # Stage 2
    applicability_model: str = ""     # Stage 6
    understanding_model: str = ""     # Stage 1 (intent, complexity, conditions of the question)
    refinement_model: str = ""        # Stage 3
    extractor_model: str = ""         # Stage C (offline profile extraction)
    fallback_model: str = ""          # the bigger model an agent escalates to on a bad outcome (empty = [llm].model)


class FeaturesConfig(BaseModel):
    orchestrator_agent: bool = True   # False: fixed rules decide the path instead of the planner agent (ablation)
    applicability_agent: bool = True  # False: the deterministic matcher alone decides coverage (ablation)
    applicability: bool = True
    contradiction: bool = True
    critic: bool = True
    profile_in_context: bool = True
    use_profiles: bool = True         # False: no Condition Profile is used online (Stage 1's name vocabulary, the answer's recorded results, the critic's check) = "the pipeline without stage C"
    escalation: bool = False         # an agent on a small model hands a bad outcome to the fallback model (see AgentsConfig)
    profile_guided_retrieval: bool = False   # Stage 6 also pulls chunks whose profiles record a missing condition
    joint_coverage: bool = False      # Stage 6 also requires model, dataset and language to be recorded TOGETHER (see applicability.py)


class Settings(BaseModel):
    llm: LLMConfig = LLMConfig()
    agents: AgentsConfig = AgentsConfig()
    models: ModelsConfig = ModelsConfig()
    devices: DevicesConfig = DevicesConfig()
    paths: PathsConfig = PathsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    applicability: ApplicabilityConfig = ApplicabilityConfig()
    contradiction: ContradictionConfig = ContradictionConfig()
    critic: CriticConfig = CriticConfig()
    features: FeaturesConfig = FeaturesConfig()


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _absolutize(paths: PathsConfig) -> PathsConfig:
    return PathsConfig(**{k: (ROOT / v if not v.is_absolute() else v) for k, v in paths.model_dump().items()})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    path = Path(os.environ.get("CGRAG_CONFIG", ROOT / "config.toml"))
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    local = ROOT / "config.local.toml"
    if local.exists():
        data = _deep_merge(data, tomllib.loads(local.read_text(encoding="utf-8")))
    settings = Settings.model_validate(data)
    settings.paths = _absolutize(settings.paths)
    return settings
