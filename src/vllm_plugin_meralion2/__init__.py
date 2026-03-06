"""vLLM plugin entrypoint and registration for MERaLiON2 models."""

from packaging.version import Version


def register() -> None:
    """Register MERaLiON2 model with vLLM's plugin system.

    Supported vLLM versions: >= 0.12.0.

    The plugin targets the V1 engine (default since vLLM 0.8.0).  A single
    adapter module (``vllm0101``) covers the entire supported range; internal
    API differences across minor versions are handled inside that module with
    defensive imports.

    Raises:
        RuntimeError: If the installed vLLM version is not supported.
    """
    from importlib import import_module

    import vllm
    from vllm import ModelRegistry

    current_version = Version(vllm.__version__)
    min_supported_version = Version("0.12.0")
    # Soft upper cap: tested up to 0.16.x.  Newer patch/minor releases are
    # expected to remain compatible; bump this when a breaking change is found.
    max_supported_version = Version("0.17.0")

    if not (min_supported_version <= current_version < max_supported_version):
        raise RuntimeError(
            f"MERaLiON2 plugin does not support vLLM version {vllm.__version__}. "
            f"Supported range: >= {min_supported_version}, < {max_supported_version}"
        )

    module = import_module("vllm_plugin_meralion2.vllm0101")
    meralion2_model_cls = getattr(module, "MERaLiON2ForConditionalGeneration")

    if "MERaLiON2ForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MERaLiON2ForConditionalGeneration",
            meralion2_model_cls,
        )
