"""Inference-only MERaLiON AudioLLM model compatible with HuggingFace weights."""
from functools import lru_cache
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData,
                         InputContext, token_inputs)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.gemma2 import Gemma2Model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.utils import consecutive_placeholder_ranges
from vllm.sequence import IntermediateTensors, SequenceData
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import maybe_prefix

from .transformers_utils.modules import (autoset_attn_implementation_for_whisper,
    MERaLiON2Inputs, MERaLiON2SpeechAudioAdaper)


logger = init_logger(__name__)


# gemma2 ties word embedding by default
_KEYS_TO_MODIFY_MAPPING = {
    "text_decoder.model": "model",
}

# === Constants === #
DEFAULT_SAMPLE_RATE = 16000
FEATURE_CHUNK_SIZE = DEFAULT_SAMPLE_RATE * 30
OUTPUT_CHUNK_SIZE = 100
MAX_NUMBER_CHUNKS = 10


def dummy_data_for_meralion(
    ctx: InputContext,
    seq_len: int,
    mm_counts: Mapping[str, int],
) -> DummyData:
    """Generate dummy data for MERaLiON model initialization.

    Args:
        ctx: Input context containing model configuration.
        seq_len: Maximum sequence length.
        mm_counts: Mapping of modality names to counts.

    Returns:
        DummyData object with dummy sequence data and audio data.

    Raises:
        RuntimeError: If the sequence length is insufficient for the audio count.
        ValueError: If audio count is invalid.
    """
    num_audios = mm_counts["audio"]
    max_tokens_per_audio = get_max_meralion_audio_tokens(ctx)
    max_llm_audio_tokens = max_tokens_per_audio * num_audios
    
    if seq_len - max_llm_audio_tokens - 2 < 0:
        raise RuntimeError(
            f"MERaLiON-AudioLLM cannot process {num_audios} audios in a prompt, "
            "please increase max_model_len or reduce audio limit by "
            "--limit-mm-per-prompt."
        )

    speech_token_index = ctx.model_config.hf_config.speech_token_index

    dummy_seqdata = SequenceData.from_prompt_token_counts(
        (speech_token_index, max_llm_audio_tokens),
        (0, seq_len - max_llm_audio_tokens),
    )
    dummy_audio = np.full(
        (MAX_NUMBER_CHUNKS * FEATURE_CHUNK_SIZE * num_audios,),
        0.0,
        dtype=np.float32,
    )
    return DummyData(
        dummy_seqdata,
        {"audio": [(dummy_audio, DEFAULT_SAMPLE_RATE)] * num_audios},
        {
            "audio": consecutive_placeholder_ranges(
                num_items=num_audios,
                item_size=max_tokens_per_audio
            )
        },
    )


def get_processor(
    processor_name: str,
    *args: object,
    trust_remote_code: bool = True, 
    **kwargs: object,
) -> object:
    """Gets a processor for the given model name via HuggingFace.

    Derived from `vllm.transformers_utils.image_processor.get_image_processor`.

    Args:
        processor_name: The name or path of the processor to load.
        *args: Additional positional arguments passed to AutoProcessor.
        trust_remote_code: Whether to trust remote code when loading the processor.
            Defaults to False for security. Set to True only if you trust the source.
        **kwargs: Additional keyword arguments passed to AutoProcessor.

    Returns:
        The loaded processor instance.

    Raises:
        RuntimeError: If the processor cannot be loaded and trust_remote_code is False.
        ValueError: If the processor cannot be loaded for other reasons.
    """
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the processor. If the processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return processor


cached_get_processor = lru_cache(get_processor)


def _get_number_chunks(audios: List[np.ndarray]) -> np.ndarray:
    """Calculate the number of chunks for each audio.

    Args:
        audios: List of audio arrays.

    Returns:
        Array of chunk counts, clipped to MAX_NUMBER_CHUNKS.
    """
    
    audio_lengths = np.array([audio.shape[0] for audio in audios])
    number_chunks = ((audio_lengths - 1) // FEATURE_CHUNK_SIZE) + 1
    return np.clip(number_chunks, a_min=1, a_max=MAX_NUMBER_CHUNKS)


def _get_chunked_audios(audios: List[np.ndarray]) -> List[np.ndarray]:
    """Split audios into chunks of FEATURE_CHUNK_SIZE.

    Args:
        audios: List of audio arrays to chunk.

    Returns:
        List of audio chunks.
    """
    if not audios:
        return []
    
    audio_number_chunks = _get_number_chunks(audios)
    chunked_resampled_audios: List[np.ndarray] = []

    for audio_idx, audio in enumerate(audios):
        for cid in range(audio_number_chunks[audio_idx]):
            chunked_resampled_audios.append(
                audio[cid * FEATURE_CHUNK_SIZE: (cid + 1) * FEATURE_CHUNK_SIZE]
            )
    return chunked_resampled_audios


def _maybe_resample_audio(
    audio: np.ndarray,
    orig_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """Resample audio if sample rates differ.

    Args:
        audio: Audio array to resample.
        orig_sample_rate: Original sample rate.
        target_sample_rate: Target sample rate.

    Returns:
        Resampled audio array if rates differ, otherwise original audio.
    """
    
    if orig_sample_rate != target_sample_rate:
        return librosa.resample(
            audio,
            orig_sr=orig_sample_rate,
            target_sr=target_sample_rate,
        )
    return audio


def get_max_meralion_audio_tokens(ctx: InputContext) -> int:
    """
    The max number of tokens after speech audio adapter.
    """
    output_chunk_size = getattr(
        ctx.model_config.hf_config, "fixed_speech_embeds_length", OUTPUT_CHUNK_SIZE)
    return MAX_NUMBER_CHUNKS * output_chunk_size


def input_processor_for_meralion(
    ctx: InputContext,
    inputs: DecoderOnlyInputs,
) -> DecoderOnlyInputs:
    """Process inputs for MERaLiON model, handling audio token replacement.

    Args:
        ctx: Input context containing model configuration.
        inputs: Decoder-only inputs dictionary.

    Returns:
        Processed inputs with audio tokens expanded.
    """
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "audio" not in multi_modal_data:
        return inputs

    audios = multi_modal_data["audio"]
    if not isinstance(audios, list):
        audios = [audios]

    if len(audios) == 0:
        return inputs

    processor = cached_get_processor(ctx.model_config.model)

    target_sample_rate = processor.feature_extractor.sampling_rate
    resampled_audios = [
        _maybe_resample_audio(
            audio=audio,
            orig_sample_rate=sampling_rate,
            target_sample_rate=target_sample_rate,
        )
        for audio, sampling_rate in audios
    ]

    output_chunk_size = getattr(
        ctx.model_config.hf_config,
        "fixed_speech_embeds_length",
        OUTPUT_CHUNK_SIZE,
    )
    audio_output_lengths = _get_number_chunks(resampled_audios) * output_chunk_size
    speech_token_index = ctx.model_config.hf_config.speech_token_index

    input_ids = inputs['prompt_token_ids']

    new_input_ids: List[int] = []
    audio_num = input_ids.count(speech_token_index)
    
    if len(audio_output_lengths) != audio_num:
        raise ValueError(
            f'The text input contains {audio_num} audio tokens, '
            f'but {len(audio_output_lengths)} audios provided'
        )
    
    start = 0
    for audio_idx in range(audio_num):
        end = input_ids.index(speech_token_index, start)
        new_input_ids.extend(input_ids[start:end])  # text part

        new_input_ids.extend([speech_token_index] * 
                             audio_output_lengths[audio_idx])
        start = end + 1
    
    new_input_ids.extend(input_ids[start:])

    return token_inputs(
        prompt_token_ids=new_input_ids,
        prompt=inputs.get('prompt'),
        multi_modal_data=multi_modal_data,
    )


def input_mapper_for_meralion(
    ctx: InputContext,
    multi_modal_data: Union[np.ndarray, List[np.ndarray]],
) -> MultiModalKwargs:
    """Input mapper for MERaLiON-AudioLLM.

    Args:
        ctx: Input context containing model configuration.
        multi_modal_data: Audio data as array or list of (audio, sample_rate) tuples.

    Returns:
        MultiModalKwargs containing processed audio features.

    Raises:
        RuntimeError: If feature extractor is not available.
        ValueError: If audio data is invalid.
    """
    if not isinstance(multi_modal_data, list):
        multi_modal_data = [multi_modal_data]

    if len(multi_modal_data) == 0:
        return MultiModalKwargs()

    processor = cached_get_processor(ctx.model_config.model)
    audio_feature_extractor = processor.feature_extractor
    if audio_feature_extractor is None:
        raise RuntimeError(
            "No HuggingFace audio_feature_extractor is available "
            "to process the audio object"
        )

    try:
        target_sample_rate = processor.feature_extractor.sampling_rate

        resampled_audios = [
            _maybe_resample_audio(
                audio=audio,
                orig_sample_rate=sampling_rate, 
                target_sample_rate=target_sample_rate, 
                )
            for audio, sampling_rate in multi_modal_data
        ]

        resampled_audios = _get_chunked_audios(resampled_audios)

        batch_data = audio_feature_extractor(
            resampled_audios,
            sampling_rate=target_sample_rate,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
            do_normalize=True,
        ).data
        batch_data["feature_attention_mask"] = batch_data.pop("attention_mask")
    except Exception:
        logger.error("Failed to process audio (%s)", multi_modal_data)
        raise

    return MultiModalKwargs(batch_data)


@INPUT_REGISTRY.register_dummy_data(dummy_data_for_meralion)
@INPUT_REGISTRY.register_input_processor(input_processor_for_meralion)
@MULTIMODAL_REGISTRY.register_input_mapper("audio",
                                           input_mapper_for_meralion)
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "audio", get_max_meralion_audio_tokens)
class MERaLiON2ForConditionalGeneration(nn.Module, SupportsMultiModal,
                                       SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]

    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        config.speech_config = autoset_attn_implementation_for_whisper(
            config.speech_config
        )
        self.speech_encoder = WhisperEncoder(config.speech_config)
        self.ln_speech = nn.LayerNorm(config.speech_config.d_model)
        self.speech_audio_adapter = MERaLiON2SpeechAudioAdaper(
            audio_hidden_size=config.speech_config.d_model,
            text_hidden_size=config.text_config.hidden_size,
            speech_mlp_scale_factor=getattr(
                config, "speech_mlp_scale_factor", 15
            ),
            speech_mlp_use_projection=getattr(
                config, "speech_mlp_use_projection", True
            ),
        )

        self.quant_config = quant_config

        self.model = Gemma2Model(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.text_config.vocab_size
        if config.text_config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.text_config.vocab_size,
                                          config.text_config.hidden_size,
                                          quant_config=quant_config)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            config.text_config.vocab_size,
            logit_scale,
        )

        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def _validate_and_reshape_mm_tensor(
        self,
        mm_input: Union[torch.Tensor, List[torch.Tensor]],
        name: str,
    ) -> torch.Tensor:
        """Validate and reshape multimodal tensor input.

        Args:
            mm_input: Input tensor or list of tensors.
            name: Name of the input for error messages.

        Returns:
            Concatenated tensor.

        Raises:
            ValueError: If input type is invalid.
        """
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(
                f"Incorrect type of {name}. "
                f"Expected torch.Tensor or list, got {type(mm_input)}"
            )

        tensors: List[torch.Tensor]
        if isinstance(mm_input, torch.Tensor):
            tensors = [mm_input]
        else:
            tensors = mm_input

        # vLLM may batch multimodal kwargs and add one or more leading batch
        # dimensions. Whisper expects:
        # - input_features: [N, mel_bins, frames]
        # - feature_attention_mask: [N, frames]
        flattened_tensors: List[torch.Tensor] = []
        for tensor in tensors:
            if name == "feature_attention_mask":
                flattened_tensors.append(tensor.reshape(-1, tensor.size(-1)))
            else:
                flattened_tensors.append(
                    tensor.reshape(-1, tensor.size(-2), tensor.size(-1))
                )

        return torch.concat(flattened_tensors, dim=0)

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> Optional[MERaLiON2Inputs]:
        """Parse and validate audio input from kwargs.

        Args:
            **kwargs: Keyword arguments containing input_features and
                feature_attention_mask.

        Returns:
            MERaLiON2Inputs if audio input is present, None otherwise.

        Raises:
            ValueError: If input format is invalid.
        """
        input_features = kwargs.pop('input_features', None)
        feature_attention_mask = kwargs.pop('feature_attention_mask', None)
        
        if input_features is None:
            return None
        input_features = self._validate_and_reshape_mm_tensor(
            input_features, 'input_features'
        )
        feature_attention_mask = self._validate_and_reshape_mm_tensor(
            feature_attention_mask, 'feature_attention_mask')
        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError("Incorrect type of audio input features. "
                             f"Got type: {type(input_features)}")
        return MERaLiON2Inputs(input_features=input_features,
                                feature_attention_mask=feature_attention_mask)

    def _process_audio_input(self,
                             audio_input: MERaLiON2Inputs) -> torch.Tensor:

        input_features = audio_input["input_features"].to(self.speech_encoder.dtype)
        feature_attention_mask = audio_input["feature_attention_mask"]

        audio_outputs = self.speech_encoder(
            input_features,
            attention_mask=feature_attention_mask,
        )
        audio_features = audio_outputs.last_hidden_state
        audio_features = self.ln_speech(audio_features)
        audio_features = self.speech_audio_adapter(audio_features)
        audio_features = audio_features.view(-1, audio_features.size(-1))

        return audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            audio_input = self._parse_and_validate_audio_input(**kwargs)

            if audio_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.model.embed_tokens(input_ids)
                processed_audio_features = self._process_audio_input(audio_input)
                # merge llm embeddings and audio features
                mask = (input_ids == self.config.speech_token_index)
                inputs_embeds[mask, :] = processed_audio_features

                input_ids = None

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Compute logits from hidden states.

        Args:
            hidden_states: Hidden states tensor.
            sampling_metadata: Sampling metadata.

        Returns:
            Logits tensor.
        """
        logits = self.logits_processor(
            self.lm_head, hidden_states, sampling_metadata
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Sample next tokens from logits.

        Args:
            logits: Logits tensor.
            sampling_metadata: Sampling metadata.

        Returns:
            SamplerOutput with next tokens.
        """
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> None:
        """Load model weights from checkpoint.

        Args:
            weights: Iterable of (name, weight) tuples.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            
            if (
                self.config.text_config.tie_word_embeddings
                and "lm_head.weight" in name
            ):
                continue
            
            # Apply key modifications
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name or 'speech_' in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)