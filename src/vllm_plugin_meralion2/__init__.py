from typing import Callable, Optional, cast

from packaging.version import Version
from vllm.entrypoints.chat_utils import BaseMultiModalItemTracker

from .transformers_utils.no_repeat_logits_processor import NoRepeatNGramLogitsProcessor


_PLACEHOLDER_ATTR = "_placeholder_str"
_original_placeholder_str: Optional[Callable[..., Optional[str]]] = getattr(
    BaseMultiModalItemTracker, _PLACEHOLDER_ATTR, None
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

    if _original_placeholder_str is None:
        return None

    original_placeholder_str = cast(
        Callable[..., Optional[str]], _original_placeholder_str
    )
    return original_placeholder_str(
        self, modality=modality, current_count=current_count
    )


def register() -> None:
    """Register MERaLiON2 model with vLLM's plugin system.

    Raises:
        RuntimeError: If vLLM version is not supported.
        ImportError: If required modules cannot be imported.
    """
    from importlib import import_module

    import vllm
    from vllm import ModelRegistry

    current_version = Version(vllm.__version__)
    min_supported_version = Version("0.8.5")
    v010_boundary = Version("0.10.0")
    max_supported_version = Version("0.11.0")

    if min_supported_version <= current_version < v010_boundary:
        module = import_module("vllm_plugin_meralion2.vllm085")
    elif v010_boundary <= current_version < max_supported_version:
        module = import_module("vllm_plugin_meralion2.vllm010")
    else:
        raise RuntimeError(
            f"MERaLiON2 doesn't support vLLM version {vllm.__version__}. "
            f"Supported vLLM versions: >= {min_supported_version}, < {max_supported_version}"
        )
    MERaLiON2ForConditionalGeneration = getattr(
        module, "MERaLiON2ForConditionalGeneration"
    )

    if "MERaLiON2ForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MERaLiON2ForConditionalGeneration",
            MERaLiON2ForConditionalGeneration,
        )

    # vLLM internals changed across versions; patch only when the target
    # attribute exists to avoid import-time/plugin-load failures.
    if hasattr(
        vllm.entrypoints.chat_utils.BaseMultiModalItemTracker, _PLACEHOLDER_ATTR
    ):
        setattr(
            vllm.entrypoints.chat_utils.BaseMultiModalItemTracker,
            _PLACEHOLDER_ATTR,
            custom_placeholder_str,
        )
