"""ASR benchmark-style integration tests for served AudioLLM."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
import soundfile as sf
from datasets import Dataset, load_from_disk
from openai import OpenAI

# Add tests directory to path if needed
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from test_utils import get_openai_client
from text_normalizer import (
    preprocess_text_asr,
    preprocess_text_asr_code_switch_chinese,
    preprocess_text_asr_malay,
    preprocess_text_asr_tamil,
)


PROMPT_TEXT = "Please transcribe this speech."
PROMPT_TEMPLATE = (
    "Instruction: {text_input} \n"
    "Follow the text instruction based on the following audio: <SpeechHere>"
)

DEFAULT_PRIVATE_DATA_ROOT = "/home/yingxu/private_data"
MAX_WORKERS = int(os.getenv("ASR_TEST_MAX_WORKERS", "12"))
ASR_CHUNK_SECONDS = 30.0
DEFAULT_DATASETS = [
    "idpc_short_ASR_v2",
    "ste_test3",
    "ytb_asr_batch1",
    "ytb_asr_batch2",
    "ytb_asr_batch3_chinese",
    "ytb_asr_batch3_malay",
    "ytb_asr_batch3_tamil_v2",
]

DEFAULT_DATASETS_WER = {
    "idpc_short_ASR_v2": 0.16,
    "ste_test3": 0.15,
    "ytb_asr_batch1": 0.11,
    "ytb_asr_batch2": 0.12,
    "ytb_asr_batch3_chinese": 0.17,
    "ytb_asr_batch3_malay": 0.18,
    "ytb_asr_batch3_tamil_v2": 0.35,
}

# Per-dataset normalizer mapping (Audiobench-compatible).
# Datasets not in this map use the lightweight _normalize_text fallback.
_DATASET_NORMALIZER = {
    "idpc_short_ASR_v2": preprocess_text_asr,
    "ste_test3": preprocess_text_asr,
    "ytb_asr_batch1": preprocess_text_asr,
    "ytb_asr_batch2": preprocess_text_asr,
    "ytb_asr_batch3_chinese": preprocess_text_asr_code_switch_chinese,
    "ytb_asr_batch3_malay": preprocess_text_asr_malay,
    "ytb_asr_batch3_tamil_v2": preprocess_text_asr_tamil,
}

# Dataset name → directory name overrides (when the directory name doesn't
# match the logical dataset name).
_DATASET_PATH_OVERRIDES = {
    "ytb_asr_batch3_tamil_v2": "ytb_asr_batch3_tamil_filtered",
}


def _normalize_text_fallback(text: str) -> str:
    """Lightweight fallback normalizer for datasets without an Audiobench normalizer."""
    text = text.lower()
    text = re.sub(r"(\[|\(|\{|\<)[^\(\)\[\]\{\}\<\>]*(\]|\)|\}|\>)", " ", text)
    text = re.sub(r"[^\w\s\u4e00-\u9fff\u0E00-\u0E7F\u0B80-\u0BFF]", " ", text)
    text = re.sub(r"\b(uh|umm|um|er|ah)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_for_wer(text: str, dataset_name: str = "") -> list[str]:
    """Normalize and tokenize text for WER computation.

    Uses per-dataset Audiobench normalizers when available, falling back to
    the lightweight regex normalizer for unknown datasets.  After normalization,
    the text is simply split on whitespace — language-specific character
    spacing (e.g. Chinese) is handled inside the normalizer.
    """
    normalizer = _DATASET_NORMALIZER.get(dataset_name, _normalize_text_fallback)
    normalized = normalizer(text)
    if not normalized:
        return []
    return normalized.split()


def _levenshtein_distance(reference: list[str], prediction: list[str]) -> int:
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)

    prev = list(range(len(prediction) + 1))
    for i, ref_tok in enumerate(reference, start=1):
        curr = [i]
        for j, pred_tok in enumerate(prediction, start=1):
            substitution_cost = 0 if ref_tok == pred_tok else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + substitution_cost,
                )
            )
        prev = curr
    return prev[-1]


def _compute_dataset_wer(
    references: Iterable[str], predictions: Iterable[str], dataset_name: str = ""
) -> float:
    total_errors = 0
    total_ref_tokens = 0

    for reference, prediction in zip(references, predictions):
        ref_tokens = _tokenize_for_wer(reference, dataset_name)
        pred_tokens = _tokenize_for_wer(prediction, dataset_name)
        total_errors += _levenshtein_distance(ref_tokens, pred_tokens)
        total_ref_tokens += len(ref_tokens)

    if total_ref_tokens == 0:
        return 0.0
    return total_errors / total_ref_tokens


def _extract_audio_and_reference(sample: dict) -> tuple[dict, str]:
    context = sample["context"]
    answer = sample["answer"]

    if isinstance(context, dict) and "audio" in context:
        audio = context["audio"]
    else:
        audio = context

    if isinstance(answer, dict) and "text" in answer:
        reference = answer["text"]
    else:
        reference = answer

    return audio, str(reference)


def _audio_to_base64_wav(audio: dict | str) -> str:
    if isinstance(audio, str):
        audio_path = Path(audio)
        audio_bytes = audio_path.read_bytes()
        return base64.b64encode(audio_bytes).decode("utf-8")

    if isinstance(audio, dict) and "array" in audio:
        array = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio.get("sampling_rate", 16000))
        buffer = io.BytesIO()
        sf.write(buffer, array, sample_rate, format="WAV")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    if isinstance(audio, dict) and "path" in audio:
        audio_path = Path(audio["path"])
        audio_bytes = audio_path.read_bytes()
        return base64.b64encode(audio_bytes).decode("utf-8")

    raise TypeError(f"Unsupported audio payload type: {type(audio)!r}")


def _load_audio_array_and_sample_rate(audio: dict | str) -> tuple[np.ndarray, int]:
    """Load audio into a mono float32 array and its sample rate."""
    if isinstance(audio, str):
        array, sample_rate = sf.read(str(audio), dtype="float32")
    elif isinstance(audio, dict) and "array" in audio:
        array = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = int(audio.get("sampling_rate", 16000))
    elif isinstance(audio, dict) and "path" in audio:
        array, sample_rate = sf.read(str(audio["path"]), dtype="float32")
    else:
        raise TypeError(f"Unsupported audio payload type: {type(audio)!r}")

    # Collapse multi-channel audio to mono for consistent chunking.
    if array.ndim > 1:
        array = np.mean(array, axis=1, dtype=np.float32)

    return np.asarray(array, dtype=np.float32), int(sample_rate)


def _chunk_audio_payload(audio: dict | str, chunk_seconds: float = ASR_CHUNK_SECONDS) -> list[dict]:
    """Split long audio into fixed-duration chunks (Audiobench style)."""
    array, sample_rate = _load_audio_array_and_sample_rate(audio)
    chunk_samples = max(1, int(chunk_seconds * sample_rate))
    if len(array) <= chunk_samples:
        return [{"array": array, "sampling_rate": sample_rate}]

    chunks: list[dict] = []
    for start in range(0, len(array), chunk_samples):
        chunks.append(
            {
                "array": array[start:start + chunk_samples],
                "sampling_rate": sample_rate,
            }
        )
    return chunks


def _message_text(response) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [part.text for part in content if hasattr(part, "text") and part.text]
        return " ".join(texts).strip()
    return str(content).strip()


def _get_dataset_names() -> list[str]:
    if os.getenv("ASR_TEST_DATASETS"):
        return [d.strip() for d in os.getenv("ASR_TEST_DATASETS", "").split(",") if d.strip()]
    return DEFAULT_DATASETS


def _transcribe_single_sample(sample: dict, client: OpenAI, model_name: str) -> tuple[str, str]:
    audio, reference = _extract_audio_and_reference(sample)
    chunk_payloads = _chunk_audio_payload(audio, chunk_seconds=ASR_CHUNK_SECONDS)

    chunk_predictions: list[str] = []
    for chunk_audio in chunk_payloads:
        audio_base64 = _audio_to_base64_wav(chunk_audio)
        content = [
            {"type": "text", "text": PROMPT_TEMPLATE.format(text_input=PROMPT_TEXT)},
            {
                "type": "audio_url",
                "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
            },
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=1024,
            temperature=0.0,
            top_p=0.9,
            extra_body={
                "repetition_penalty": 1.0,
                "top_k": 50,
                "length_penalty": 1.0,
            },
            stream=False,
            seed=42,
        )
        chunk_predictions.append(_message_text(response))

    prediction = " ".join(part.strip() for part in chunk_predictions if part and part.strip())
    return prediction.strip(), reference


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_model
@pytest.mark.parametrize("dataset_name", _get_dataset_names())
def test_served_audiollm_asr_wer_lt_one(dataset_name: str) -> None:
    """Evaluate transcription quality on local HF ASR datasets (WER < 1)."""
    root = Path(os.getenv("ASR_TEST_DATA_ROOT", DEFAULT_PRIVATE_DATA_ROOT))
    dir_name = _DATASET_PATH_OVERRIDES.get(dataset_name, dataset_name)
    dataset_path = root / dir_name
    if not dataset_path.exists():
        pytest.skip(f"Dataset path not found: {dataset_path}")

    raw_data = load_from_disk(str(dataset_path))
    # raw_data = raw_data.filter(lambda examples: [ 2 < al < 120 for al in examples["audio_length"]], batched=True)
    if isinstance(raw_data, dict):
        # Defensive fallback for DatasetDict-like payloads
        split_name = "test" if "test" in raw_data else next(iter(raw_data.keys()))
        data = raw_data[split_name]
    else:
        data = raw_data

    assert isinstance(data, Dataset)

    max_samples = int(os.getenv("ASR_TEST_MAX_SAMPLES", "0"))
    if max_samples > 0 and len(data) > max_samples:
        data = data.select(range(max_samples))

    client, model_name = get_openai_client()
    samples = list(data)
    if not samples:
        pytest.skip(f"Dataset is empty: {dataset_name}")

    workers = min(MAX_WORKERS, len(samples))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda sample: _transcribe_single_sample(sample, client, model_name),
                samples,
            )
        )

    predictions = [prediction for prediction, _ in results]
    references = [reference for _, reference in results]

    dataset_wer = _compute_dataset_wer(references, predictions, dataset_name)
    wer_threshold = DEFAULT_DATASETS_WER[dataset_name]

    print(f"{dataset_name} WER={dataset_wer:.4f}")
    assert dataset_wer < wer_threshold, f"{dataset_name} WER={dataset_wer:.4f} (expected < {wer_threshold:.4f})"
