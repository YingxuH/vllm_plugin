"""vLLM plugin entrypoint and registration for MERaLiON2 models."""

from packaging.version import Version


def _patch_logitsprocs_output_token_tracking() -> None:
    """Fix vLLM bug: entry-point logits processors don't enable output token tracking.

    vLLM sets ``logitsprocs_need_output_token_ids`` based on CLI-passed
    ``custom_logitsprocs`` only, ignoring processors discovered via the
    ``vllm.logits_processors`` entry-point group.  When this flag is False
    and ``repetition_penalty=1.0`` (no penalties), the output-token live
    reference is filled with ``-1`` placeholders instead of actual token IDs,
    silently disabling any processor that inspects generation history
    (e.g. ``NoRepeatNGramV1LogitsProcessor``).

    This patch wraps ``InputBatch.__init__`` to force the flag when
    non-argmax-invariant entry-point processors are loaded.
    """
    try:
        from vllm.v1.worker.gpu_input_batch import InputBatch
    except ImportError:
        return  # vLLM version without V1 InputBatch

    _orig_init = InputBatch.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # If entry-point plugins loaded non-argmax-invariant processors,
        # they need output_token_ids to be tracked — force the flag.
        if (hasattr(self, "logitsprocs")
                and self.logitsprocs.non_argmax_invariant
                and not self.logitsprocs_need_output_token_ids):
            self.logitsprocs_need_output_token_ids = True

    InputBatch.__init__ = _patched_init


def _patch_streaming_asr_endpoint() -> None:
    """Monkey-patch vLLM's ``build_app`` to include the streaming ASR endpoint.

    The patch wraps the original ``build_app`` function so that after
    the FastAPI app is created with all standard routes, the streaming
    ASR WebSocket route (``/v1/streaming_asr``) is attached.

    This runs inside the vLLM process, sharing the engine and GPU with
    ``/v1/chat/completions``.  The existing endpoint is unaffected.
    """
    try:
        from vllm.entrypoints.openai import api_server
    except ImportError:
        return  # vLLM not installed or incompatible version

    _orig_build_app = getattr(api_server, "build_app", None)
    if _orig_build_app is None:
        return  # vLLM version without build_app

    def _patched_build_app(args, **kwargs):
        app = _orig_build_app(args, **kwargs)
        try:
            from vllm_plugin_meralion2.streaming.endpoint import (
                register_streaming_asr,
                init_streaming_state,
            )
            register_streaming_asr(app)
            # Extract model name from args for the streaming endpoint
            model_name = getattr(args, "served_model_name", None)
            if isinstance(model_name, list) and model_name:
                model_name = model_name[0]
            elif model_name is None:
                model_name = getattr(args, "model", "")
            init_streaming_state(app, model_name=model_name or "")
        except Exception as exc:
            import logging
            logging.getLogger("vllm_plugin").warning(
                "Failed to register streaming ASR endpoint: %s", exc,
            )
        return app

    api_server.build_app = _patched_build_app


def register() -> None:
    """Register MERaLiON2 model with vLLM's plugin system.

    Supported vLLM versions: >= 0.12.0, < 0.17.0.

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
    # Tested range: 0.12.0 – 0.16.x.  The InputBatch monkey-patch in
    # _patch_logitsprocs_output_token_tracking() fixes a vLLM V1 bug where
    # entry-point logits processors don't receive output token IDs, enabling
    # NoRepeatNGram to work across all supported versions with rep_penalty=1.0.
    max_supported_version = Version("0.17.0")

    if min_supported_version > current_version or current_version >= max_supported_version:
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

    _patch_logitsprocs_output_token_tracking()
    _patch_streaming_asr_endpoint()
