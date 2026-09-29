"""Stage 1's classifier: SciBERT with two heads - intent (5 classes) and complexity (2 classes).

The architecture specifies one SciBERT model with two heads. It is trained by scripts/train_query_classifier.py and saved
under data/index/query_classifier/; QueryUnderstanding uses it when that checkpoint exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

import torch
from torch import nn

from ..config import get_settings
from ..schemas import Complexity, Intent

INTENTS: tuple[str, ...] = get_args(Intent)
COMPLEXITIES: tuple[str, ...] = get_args(Complexity)
MAX_LEN = 96


def checkpoint_dir() -> Path:
    return get_settings().paths.index_dir / "query_classifier"


class TwoHeadClassifier(nn.Module):
    def __init__(self, encoder_name: str) -> None:
        from transformers import AutoModel

        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.intent_head = nn.Linear(hidden, len(INTENTS))
        self.complexity_head = nn.Linear(hidden, len(COMPLEXITIES))

    def forward(self, **enc) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.dropout(self.encoder(**enc).last_hidden_state[:, 0])   # [CLS]
        return self.intent_head(pooled), self.complexity_head(pooled)


@dataclass
class Prediction:
    intent: str
    complexity: str
    intent_prob: float
    complexity_prob: float


class IntentClassifier:
    """Inference wrapper around a trained TwoHeadClassifier."""

    def __init__(self, model: TwoHeadClassifier, tokenizer, device: str) -> None:
        self.model, self.tokenizer, self.device = model.to(device).eval(), tokenizer, device

    @classmethod
    def load(cls, directory: Path | None = None) -> "IntentClassifier | None":
        """The trained classifier, or None when no checkpoint has been trained yet."""
        from transformers import AutoTokenizer

        directory = directory or checkpoint_dir()
        meta = directory / "meta.json"
        if not (directory / "weights.pt").exists() or not meta.exists():
            return None
        encoder = json.loads(meta.read_text())["encoder"]
        model = TwoHeadClassifier(encoder)
        model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu"))
        return cls(model, AutoTokenizer.from_pretrained(encoder), get_settings().devices.classifier)

    @torch.inference_mode()
    def predict(self, question: str) -> Prediction:
        enc = self.tokenizer(question, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(self.device)
        intent_logits, complexity_logits = self.model(**enc)
        ip, cp = intent_logits.softmax(-1)[0], complexity_logits.softmax(-1)[0]
        i, c = int(ip.argmax()), int(cp.argmax())
        return Prediction(INTENTS[i], COMPLEXITIES[c], float(ip[i]), float(cp[c]))


def save(model: TwoHeadClassifier, encoder_name: str, directory: Path | None = None) -> Path:
    directory = directory or checkpoint_dir()
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "weights.pt")
    (directory / "meta.json").write_text(json.dumps({"encoder": encoder_name, "intents": INTENTS, "complexities": COMPLEXITIES}))
    return directory
