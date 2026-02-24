"""Adapter for vLLM 0.10.x built on top of the 0.8/0.9 implementation."""

# The 0.10.x lane keeps a dedicated module boundary so future API-specific
# shims can be implemented without affecting the 0.8/0.9 adapter line.
from .vllm085 import (  # noqa: F401
    MERaLiON2ForConditionalGeneration as _MERaLiON2ForConditionalGeneration085,
)


class MERaLiON2ForConditionalGeneration(_MERaLiON2ForConditionalGeneration085):
    """vLLM 0.10.x adapter class for MERaLiON2."""
