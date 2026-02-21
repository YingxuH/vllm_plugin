from typing import Optional

from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker

from .transformers_utils.no_repeat_logits_processor import NoRepeatNGramLogitsProcessor


_original_placeholder_str = getattr(
    BaseMultiModalItemTracker, "_placeholder_str"
)


def custom_placeholder_str(
    self,
    modality: str,
    current_count: int,
) -> Optional[str]:
    """Custom placeholder string for MERaLiON2 audio modality.

    Args:
        modality: The modality name (e.g., "audio").
        current_count: Current count of items.

    Returns:
        Placeholder string for audio modality, or None to use default.
    """
    hf_config = self._model_config.hf_config
    model_type = hf_config.model_type

    if modality == "audio" and model_type == "meralion2":
        return "<SpeechHere>"

    return _original_placeholder_str(
        self, modality=modality, current_count=current_count
    )


def register() -> None:
    """Register MERaLiON2 model with vLLM's plugin system.

    Raises:
        RuntimeError: If vLLM version is not supported.
        ImportError: If required modules cannot be imported.
    """
    import vllm
    from vllm import ModelRegistry

    v064_compatible_versions = [
        "0.6.5",
        "0.6.6",
        "0.6.6.post1",
        "0.7.0",
        "0.7.1",
        "0.7.2",
        "0.7.3",
    ]
    v085_compatible_versions = ["0.8.5", "0.8.5.post1"]
    sorted_compatible_versions = sorted(
        v064_compatible_versions + v085_compatible_versions
    )

    if vllm.__version__ in v064_compatible_versions:
        from .vllm064_post1 import MERaLiON2ForConditionalGeneration
    elif vllm.__version__ in v085_compatible_versions:
        from .vllm085 import MERaLiON2ForConditionalGeneration
    else:
        raise RuntimeError(
            f"MERaLiON2 doesn't support vLLM version {vllm.__version__}. "
            f"Supported vLLM versions: {', '.join(sorted_compatible_versions)}"
        )

    if "MERaLiON2ForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MERaLiON2ForConditionalGeneration",
            MERaLiON2ForConditionalGeneration,
        )

    setattr(
        vllm.entrypoints.chat_utils.BaseMultiModalItemTracker,
        "_placeholder_str",
        custom_placeholder_str,
    )
