"""Latency integration tests for served AudioLLM (TTFT and ITL)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os
import sys
from pathlib import Path
import statistics
import time
from typing import Any

from datasets import Dataset, load_from_disk
from openai import OpenAI
import pytest

# Add tests directory to path if needed
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from test_asr_transcription_eval import (
    DEFAULT_PRIVATE_DATA_ROOT,
    MAX_WORKERS,
    PROMPT_TEMPLATE,
    PROMPT_TEXT,
    _audio_to_base64_wav,
    _extract_audio_and_reference,
    _get_dataset_names,
)
from test_utils import get_openai_client

TTFT_P95_BUDGET_MS = 600
ITL_P95_BUDGET_MS = 50
DEFAULT_MAX_SAMPLES = 32
DEFAULT_MAX_COMPLETION_TOKENS = 256


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer. Got: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"Environment variable {name} must be > 0. Got: {parsed}")
    return parsed


def _get_optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float. Got: {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"Environment variable {name} must be > 0. Got: {parsed}")
    return parsed


def _percentile(values: list[float], ratio: float) -> float:
    """Simple nearest-rank percentile."""
    if not values:
        raise ValueError("Cannot compute percentile from empty values")
    ordered = sorted(values)
    rank = max(1, int(math.ceil(ratio * len(ordered))))
    return ordered[rank - 1]


def _iter_stream_text_tokens(stream: Any):
    """Yield decoded text chunks from OpenAI streaming responses."""
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if not content:
            continue

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(part.text for part in content if hasattr(part, "text") and part.text)
        else:
            text = str(content)

        if text:
            yield text


def _stream_single_sample(sample: dict, client: OpenAI, model_name: str, max_completion_tokens: int) -> dict[str, Any]:
    audio, _reference = _extract_audio_and_reference(sample)
    audio_base64 = _audio_to_base64_wav(audio)
    content = [
        {"type": "text", "text": PROMPT_TEMPLATE.format(text_input=PROMPT_TEXT)},
        {
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
        },
    ]

    start_t = time.perf_counter()
    stream = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=max_completion_tokens,
        temperature=0.0,
        top_p=0.9,
        extra_body={
            "repetition_penalty": 1.0,
            "top_k": 50,
            "length_penalty": 1.0,
            "logits_processors": [
                {"qualname": "vllm_plugin_meralion2.NoRepeatNGramLogitsProcessor", "args": [6]}
            ],
        },
        stream=True,
        seed=42,
    )

    first_token_t: float | None = None
    token_arrival_times: list[float] = []
    for _token_text in _iter_stream_text_tokens(stream):
        now = time.perf_counter()
        if first_token_t is None:
            first_token_t = now
        token_arrival_times.append(now)
    end_t = time.perf_counter()

    if first_token_t is None:
        return {
            "ttft_ms": None,
            "itl_ms_list": [],
            "tokens": 0,
            "e2e_ms": (end_t - start_t) * 1000,
        }

    itl_ms_list = [
        (token_arrival_times[i] - token_arrival_times[i - 1]) * 1000
        for i in range(1, len(token_arrival_times))
    ]
    return {
        "ttft_ms": (first_token_t - start_t) * 1000,
        "itl_ms_list": itl_ms_list,
        "tokens": len(token_arrival_times),
        "e2e_ms": (end_t - start_t) * 1000,
    }


def _load_dataset(dataset_name: str) -> Dataset:
    root = Path(os.getenv("ASR_TEST_DATA_ROOT", DEFAULT_PRIVATE_DATA_ROOT))
    dataset_path = root / dataset_name
    if not dataset_path.exists():
        pytest.skip(f"Dataset path not found: {dataset_path}")

    raw_data = load_from_disk(str(dataset_path))
    if isinstance(raw_data, dict):
        split_name = "test" if "test" in raw_data else next(iter(raw_data.keys()))
        data = raw_data[split_name]
    else:
        data = raw_data

    assert isinstance(data, Dataset)
    return data


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_model
@pytest.mark.parametrize("dataset_name", _get_dataset_names())
def test_served_audiollm_asr_latency_16_concurrent(dataset_name: str) -> None:
    """Measure TTFT and ITL under concurrent streaming requests."""
    data = _load_dataset(dataset_name)
    samples = list(data)
    if not samples:
        pytest.skip(f"Dataset is empty: {dataset_name}")

    concurrency = _get_int_env("ASR_LATENCY_CONCURRENCY", MAX_WORKERS)
    max_samples = _get_int_env("ASR_LATENCY_MAX_SAMPLES", DEFAULT_MAX_SAMPLES)
    max_completion_tokens = _get_int_env(
        "ASR_LATENCY_MAX_COMPLETION_TOKENS",
        DEFAULT_MAX_COMPLETION_TOKENS,
    )
    ttft_p95_budget_ms = os.getenv("ASR_TTFT_P95_BUDGET_MS", TTFT_P95_BUDGET_MS)
    itl_p95_budget_ms = os.getenv("ASR_ITL_P95_BUDGET_MS", ITL_P95_BUDGET_MS)

    request_count = min(len(samples), max(max_samples, concurrency))
    eval_samples = samples[:request_count]
    workers = min(concurrency, len(eval_samples))

    client, model_name = get_openai_client()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda sample: _stream_single_sample(
                    sample=sample,
                    client=client,
                    model_name=model_name,
                    max_completion_tokens=max_completion_tokens,
                ),
                eval_samples,
            )
        )

    successful = [result for result in results if result["ttft_ms"] is not None]
    assert successful, "No streaming response emitted tokens for latency measurement."

    ttft_values = [float(result["ttft_ms"]) for result in successful]
    itl_values = [itl for result in successful for itl in result["itl_ms_list"]]
    e2e_values = [float(result["e2e_ms"]) for result in successful]
    token_counts = [int(result["tokens"]) for result in successful]

    ttft_p50 = statistics.median(ttft_values)
    ttft_p95 = _percentile(ttft_values, 0.95)

    itl_p50 = statistics.median(itl_values) if itl_values else 0.0
    itl_p95 = _percentile(itl_values, 0.95) if itl_values else 0.0
    itl_mean = statistics.fmean(itl_values) if itl_values else 0.0

    print(
        (
            f"{dataset_name} latency: "
            f"requests={len(results)} successful={len(successful)} workers={workers} "
            f"tokens_mean={statistics.fmean(token_counts):.2f} "
            f"ttft_ms(p50/p95)=({ttft_p50:.2f}/{ttft_p95:.2f}) "
            f"itl_ms(mean/p50/p95)=({itl_mean:.2f}/{itl_p50:.2f}/{itl_p95:.2f}) "
            f"e2e_ms_mean={statistics.fmean(e2e_values):.2f}"
        )
    )

    if ttft_p95_budget_ms is not None:
        assert ttft_p95 <= ttft_p95_budget_ms, (
            f"{dataset_name} TTFT p95={ttft_p95:.2f}ms exceeds budget "
            f"{ttft_p95_budget_ms:.2f}ms"
        )
    if itl_p95_budget_ms is not None:
        assert itl_p95 <= itl_p95_budget_ms, (
            f"{dataset_name} ITL p95={itl_p95:.2f}ms exceeds budget "
            f"{itl_p95_budget_ms:.2f}ms"
        )
