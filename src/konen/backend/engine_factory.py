from collections.abc import Callable
from typing import Any, ClassVar

from .claude_engine import ClaudeEngine
from .gemini_engine import GeminiEngine
from .gpt_engine import GPTEngine
from .kimi_engine import KimiEngine
from .llm_engine import LLMEngine


class EngineFactory:
    """Factory for provider-specific execution engines."""

    _builders: ClassVar[dict[str, Callable[[dict[str, Any]], LLMEngine]]] = {}

    @classmethod
    def register(
        cls, model_family: str, builder: Callable[[dict[str, Any]], LLMEngine]
    ) -> None:
        """Register a builder for a model family."""
        cls._builders[model_family] = builder

    @classmethod
    def build(
        cls,
        model_family: str,
        model_name: str,
        config: dict[str, Any] | None = None,
        uid: str | None = None,
    ) -> LLMEngine:
        """Build an engine for a model family."""
        try:
            builder = cls._builders[model_family]
        except KeyError as exc:
            raise ValueError(f"Unsupported model_family: '{model_family}'.") from exc
        kwargs: dict[str, Any] = {
            "model_family": model_family,
            "model_name": model_name,
        }
        if config is not None:
            kwargs["config"] = dict(config)
        if uid is not None:
            kwargs["uid"] = uid
        return builder(kwargs)

    @staticmethod
    def _validate_builder_kwargs(kwargs: dict[str, Any]) -> None:
        """Validate keyword arguments for an engine builder."""
        allowed_keys = {"model_family", "model_name", "config", "uid"}
        unexpected_keys = sorted(set(kwargs) - allowed_keys)
        if unexpected_keys:
            unexpected_keys_str = ", ".join(unexpected_keys)
            raise ValueError(f"Unexpected engine kwargs: {unexpected_keys_str}.")

    @staticmethod
    def _claude_engine_builder(kwargs: dict[str, Any]) -> LLMEngine:
        kwargs = dict(kwargs)
        kwargs["model_family"] = "claude"
        EngineFactory._validate_builder_kwargs(kwargs)
        return ClaudeEngine(**kwargs)

    @staticmethod
    def _gemini_engine_builder(kwargs: dict[str, Any]) -> LLMEngine:
        kwargs = dict(kwargs)
        kwargs["model_family"] = "gemini"
        EngineFactory._validate_builder_kwargs(kwargs)
        return GeminiEngine(**kwargs)

    @staticmethod
    def _gpt_engine_builder(kwargs: dict[str, Any]) -> LLMEngine:
        kwargs = dict(kwargs)
        kwargs["model_family"] = "gpt"
        EngineFactory._validate_builder_kwargs(kwargs)
        return GPTEngine(**kwargs)

    @staticmethod
    def _kimi_engine_builder(kwargs: dict[str, Any]) -> LLMEngine:
        kwargs = dict(kwargs)
        kwargs["model_family"] = "kimi"
        EngineFactory._validate_builder_kwargs(kwargs)
        return KimiEngine(**kwargs)


EngineFactory.register("claude", EngineFactory._claude_engine_builder)
EngineFactory.register("gemini", EngineFactory._gemini_engine_builder)
EngineFactory.register("gpt", EngineFactory._gpt_engine_builder)
EngineFactory.register("kimi", EngineFactory._kimi_engine_builder)
