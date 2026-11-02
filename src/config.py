"""Central configuration for the LLM Disagreement Analysis pipeline.

All tunable parameters live here so that every module receives the same,
validated settings (STD state: CONFIGURING).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root = parent of src/.  No absolute paths are hard-coded anywhere.
ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """Raised when settings are invalid (STD guard: [settings invalid])."""


@dataclass
class Config:
    # ---- Data Collection ----
    data_path: Path = ROOT / "data" / "sample_dataset.csv"
    limit: int | None = None              # use only the first N items (debugging)

    # ---- Multi-Response Sampling ----
    # Models are "provider:model-id" (e.g. "openai:<id>", "gemini:<id>", "anthropic:<id>",
    # "ollama:<id>", "mock:<name>").  A bare name uses `backend` as its provider (legacy CLI).
    backend: str = "mock"
    models: list[str] = field(default_factory=lambda: ["model-A", "model-B", "model-C"])
    n_samples: int = 5                    # responses per (prompt, model)
    temperature: float = 1.0              # must be > 0 so repeated sampling can differ
    max_tokens: int = 512
    max_retries: int = 3
    retry_base_delay: float = 0.5         # seconds; doubles each retry (rate limits honour Retry-After)
    max_requests: int = 5000              # safety cap: items x models x samples
    api_keys: dict = field(default_factory=dict)   # provider -> key typed for THIS run (memory only)

    # ---- Semantic comparison (STD state EVALUATING) ----
    judge_model: str | None = None        # "provider:model-id"; None = lexical checks only
    wording_threshold: float = 0.8        # token-overlap above which answers are "same, different wording"
    use_cache: bool = True
    seed: int = 42                        # only used by the mock backend

    # ---- Disagreement Detection ----
    agreement_threshold: float = 0.75     # agreement < threshold  => item is flagged
    jaccard_threshold: float = 0.5        # open-ended answers: token-overlap clustering
    human_agreement_threshold: float = 0.6   # humans "disagree" below this top-label share
    llm_overconfidence_threshold: float = 0.8  # LLM "confident" at/above this agreement

    # ---- Reporting ----
    output_dir: Path = ROOT / "outputs"
    top_k_uncertain: int = 10

    def validate(self) -> None:
        errs = []
        if self.backend not in {"mock", "openai", "gemini", "anthropic", "ollama", "ollama-cloud"}:
            errs.append(f"backend must be mock|openai|gemini|anthropic|ollama|ollama-cloud, got '{self.backend}'")
        if not self.models:
            errs.append("at least one model is required")
        if self.n_samples < 1:
            errs.append("n_samples must be >= 1")
        if self.n_samples > 1 and self.temperature <= 0:
            errs.append("temperature must be > 0 when n_samples > 1, otherwise repeated samples are identical")
        if not 0 <= self.temperature <= 2:
            errs.append("temperature must be between 0 and 2")
        for name in ("agreement_threshold", "human_agreement_threshold",
                     "llm_overconfidence_threshold", "jaccard_threshold", "wording_threshold"):
            v = getattr(self, name)
            if not 0 < v <= 1:
                errs.append(f"{name} must be in (0, 1], got {v}")
        if self.max_retries < 1:
            errs.append("max_retries must be >= 1")
        if errs:
            raise ConfigError("; ".join(errs))
