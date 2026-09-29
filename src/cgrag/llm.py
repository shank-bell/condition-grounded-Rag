from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from .config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    gen_tokens: int
    wall_s: float

    @property
    def tokens_per_s(self) -> float:
        return self.gen_tokens / self.wall_s if self.wall_s else 0.0


class OllamaLLM:
    def __init__(self, settings: Settings | None = None):
        self.cfg = (settings or get_settings()).llm
        self.client = ollama.Client(host=self.cfg.host)

    def chat(
        self,
        messages: list[dict],
        *,
        schema: dict | None = None,
        temperature: float | None = None,
        think: bool | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        options: dict = {
            "temperature": self.cfg.generate_temperature if temperature is None else temperature,
            "num_ctx": self.cfg.num_ctx,
        }
        if max_tokens:
            options["num_predict"] = max_tokens
        t0 = time.perf_counter()
        resp = self.client.chat(
            model=self.cfg.model,
            messages=messages,
            format=schema,
            options=options,
            think=self.cfg.think if think is None else think,
            keep_alive=self.cfg.keep_alive,
        )
        wall = time.perf_counter() - t0
        return LLMResult(
            text=resp.message.content or "",
            prompt_tokens=resp.prompt_eval_count or 0,
            gen_tokens=resp.eval_count or 0,
            wall_s=wall,
        )

    def structured(
        self,
        prompt: str,
        model_cls: type[T],
        *,
        system: str | None = None,
        temperature: float | None = None,
        retries: int = 1,
    ) -> tuple[T, LLMResult]:
        """Ask for JSON matching model_cls; retry once on invalid output."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        temp = self.cfg.extract_temperature if temperature is None else temperature
        schema = model_cls.model_json_schema()
        last_err: Exception | None = None
        for _ in range(retries + 1):
            result = self.chat(messages, schema=schema, temperature=temp)
            try:
                return model_cls.model_validate_json(result.text), result
            except ValidationError as err:
                last_err = err
        raise ValueError(f"LLM did not return valid {model_cls.__name__}: {last_err}")
