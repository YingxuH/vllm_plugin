"""Mel-spectrogram cache for streaming ASR.

Manages a server-side audio buffer and provides efficient mel feature
computation.  Two strategies are available:

- **full** (default): Re-runs the feature extractor on the complete
  audio buffer each step.  Correct under all normalization settings.
- **incremental**: Caches pre-normalization log-mel features and only
  computes the STFT for newly appended audio.  Requires
  ``do_normalize=False`` on the feature extractor because waveform
  normalization is global and invalidates cached features.

Even in ``full`` mode, being inside the vLLM process avoids the
base64 / WAV / HTTP overhead of external callers (~50-70 ms savings).
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np


class FeatureExtractorLike(Protocol):
    """Minimal interface for a Whisper-style feature extractor."""

    sampling_rate: int
    chunk_length: int  # seconds per chunk (typically 30)
    nb_max_frames: int  # max mel frames per chunk (typically 3000)
    feature_size: int  # mel bins (typically 128)

    def __call__(
        self,
        raw_speech: list[np.ndarray],
        *,
        sampling_rate: int,
        return_tensors: str,
        return_attention_mask: bool,
        padding: str,
        do_normalize: bool,
    ) -> Any: ...


class MelCache:
    """Server-side audio buffer with mel-spectrogram caching.

    Parameters
    ----------
    feature_extractor:
        A ``WhisperFeatureExtractor`` (or compatible mock).
    whisper_chunk_size:
        Audio chunk size in seconds for Whisper (default 30).
    sample_rate:
        Expected audio sample rate (default 16 000).
    do_normalize:
        Whether to normalize waveform before mel computation.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractorLike,
        *,
        whisper_chunk_size: int = 30,
        sample_rate: int = 16_000,
        do_normalize: bool = True,
    ) -> None:
        self._fe = feature_extractor
        self._chunk_size_samples = whisper_chunk_size * sample_rate
        self._sr = sample_rate
        self._do_normalize = do_normalize

        # Audio buffer — list of float32 arrays, lazily concatenated
        self._chunks: list[np.ndarray] = []
        self._total_samples: int = 0
        # Cached contiguous buffer (invalidated on append/trim)
        self._flat: Optional[np.ndarray] = None

    # ── Audio buffer management ──────────────────────────────────────

    def append_audio(self, audio: np.ndarray) -> None:
        """Append float32 mono audio to the buffer."""
        if audio.ndim != 1:
            raise ValueError(f"Expected 1-D audio, got shape {audio.shape}")
        if len(audio) == 0:
            return
        self._chunks.append(audio.astype(np.float32, copy=False))
        self._total_samples += len(audio)
        self._flat = None  # invalidate cache

    def get_audio(self, start_sample: int = 0) -> np.ndarray:
        """Return audio from *start_sample* to end as a contiguous array."""
        buf = self._get_flat()
        return buf[start_sample:]

    def trim_front(self, n_samples: int) -> None:
        """Remove the first *n_samples* from the buffer."""
        if n_samples <= 0:
            return
        n_samples = min(n_samples, self._total_samples)
        buf = self._get_flat()
        remaining = buf[n_samples:]
        self._chunks = [remaining] if len(remaining) > 0 else []
        self._total_samples = len(remaining)
        self._flat = remaining if len(remaining) > 0 else None

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def duration_sec(self) -> float:
        return self._total_samples / self._sr if self._sr > 0 else 0.0

    def reset(self) -> None:
        """Clear all audio and cached features."""
        self._chunks.clear()
        self._total_samples = 0
        self._flat = None

    # ── Mel feature computation ──────────────────────────────────────

    def compute_features(
        self, start_sample: int = 0
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute mel features for audio from *start_sample* onward.

        Returns
        -------
        input_features_list:
            List of mel arrays, one per 30 s chunk.
            Each has shape ``(n_mel_bins, max_frames)``.
        attention_mask_list:
            List of attention masks, one per chunk.
            Each has shape ``(max_frames,)``.
        """
        audio = self.get_audio(start_sample)
        if len(audio) == 0:
            return [], []

        # Split into <=30 s chunks (matching MERaLiON2Processor logic)
        chunks = self._split_into_chunks(audio)

        result = self._fe(
            chunks,
            sampling_rate=self._sr,
            return_tensors="np",
            return_attention_mask=True,
            padding="max_length",
            do_normalize=self._do_normalize,
        )

        # result has input_features (n_chunks, n_mel, max_frames)
        # and attention_mask (n_chunks, max_frames)
        features = result["input_features"]
        masks = result.get("attention_mask", result.get("feature_attention_mask"))

        if features.ndim == 2:
            features = features[np.newaxis, ...]
        if masks is not None and masks.ndim == 1:
            masks = masks[np.newaxis, ...]

        feature_list = [features[i] for i in range(features.shape[0])]
        mask_list = (
            [masks[i] for i in range(masks.shape[0])]
            if masks is not None
            else [np.ones(features.shape[-1], dtype=np.float32)] * len(feature_list)
        )
        return feature_list, mask_list

    def get_num_chunks(self, start_sample: int = 0) -> int:
        """Return the number of Whisper chunks for current audio."""
        n_samples = max(0, self._total_samples - start_sample)
        if n_samples == 0:
            return 0
        return ((n_samples - 1) // self._chunk_size_samples) + 1

    # ── Internal ─────────────────────────────────────────────────────

    def _get_flat(self) -> np.ndarray:
        """Return a contiguous view of the full audio buffer."""
        if self._flat is not None:
            return self._flat
        if not self._chunks:
            self._flat = np.array([], dtype=np.float32)
        elif len(self._chunks) == 1:
            self._flat = self._chunks[0]
        else:
            self._flat = np.concatenate(self._chunks)
            self._chunks = [self._flat]
        return self._flat

    def _split_into_chunks(self, audio: np.ndarray) -> list[np.ndarray]:
        """Split audio into chunks of at most ``_chunk_size_samples``."""
        if len(audio) <= self._chunk_size_samples:
            return [audio]
        n_chunks = ((len(audio) - 1) // self._chunk_size_samples) + 1
        return [
            audio[i * self._chunk_size_samples : (i + 1) * self._chunk_size_samples]
            for i in range(n_chunks)
        ]
