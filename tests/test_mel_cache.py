"""Unit tests for mel_cache.py — server-side audio buffer and mel computation."""

import numpy as np
import pytest

from vllm_plugin_meralion2.streaming.mel_cache import MelCache


# ── Mock feature extractor ───────────────────────────────────────────


class MockFeatureExtractor:
    """Minimal mock of WhisperFeatureExtractor for testing.

    Returns mel features with shape (n_chunks, n_mel_bins, max_frames)
    where values encode positional information for verification.
    """

    sampling_rate = 16_000
    chunk_length = 30  # seconds
    nb_max_frames = 3000
    feature_size = 128  # mel bins

    def __init__(self, *, max_frames: int = 3000, n_mel: int = 128):
        self.nb_max_frames = max_frames
        self.feature_size = n_mel
        self.call_count = 0
        self.last_chunks = None

    def __call__(
        self,
        raw_speech,
        *,
        sampling_rate=16_000,
        return_tensors="np",
        return_attention_mask=True,
        padding="max_length",
        do_normalize=True,
    ):
        self.call_count += 1
        self.last_chunks = raw_speech
        n_chunks = len(raw_speech)
        features = np.zeros(
            (n_chunks, self.feature_size, self.nb_max_frames), dtype=np.float32
        )
        masks = np.zeros((n_chunks, self.nb_max_frames), dtype=np.float32)

        for i, chunk in enumerate(raw_speech):
            n_frames = min(
                len(chunk) * self.nb_max_frames // (self.chunk_length * sampling_rate),
                self.nb_max_frames,
            )
            n_frames = max(n_frames, 1)
            features[i, :, :n_frames] = 1.0  # mark active frames
            masks[i, :n_frames] = 1.0

        return {"input_features": features, "attention_mask": masks}


@pytest.fixture
def fe():
    return MockFeatureExtractor()


@pytest.fixture
def cache(fe):
    return MelCache(fe, whisper_chunk_size=30, sample_rate=16_000)


# ── Audio buffer tests ───────────────────────────────────────────────


class TestAudioBuffer:
    def test_empty_buffer(self, cache):
        assert cache.total_samples == 0
        assert cache.duration_sec == 0.0
        audio = cache.get_audio()
        assert len(audio) == 0

    def test_append_single_chunk(self, cache):
        chunk = np.random.randn(8000).astype(np.float32)
        cache.append_audio(chunk)
        assert cache.total_samples == 8000
        assert abs(cache.duration_sec - 0.5) < 1e-6

    def test_append_multiple_chunks(self, cache):
        for _ in range(4):
            cache.append_audio(np.zeros(8000, dtype=np.float32))
        assert cache.total_samples == 32000
        assert abs(cache.duration_sec - 2.0) < 1e-6

    def test_get_audio_contiguous(self, cache):
        c1 = np.ones(100, dtype=np.float32)
        c2 = np.ones(200, dtype=np.float32) * 2.0
        cache.append_audio(c1)
        cache.append_audio(c2)
        audio = cache.get_audio()
        assert len(audio) == 300
        np.testing.assert_array_equal(audio[:100], 1.0)
        np.testing.assert_array_equal(audio[100:], 2.0)

    def test_get_audio_with_offset(self, cache):
        cache.append_audio(np.arange(100, dtype=np.float32))
        audio = cache.get_audio(start_sample=50)
        assert len(audio) == 50
        np.testing.assert_array_equal(audio, np.arange(50, 100, dtype=np.float32))

    def test_get_audio_offset_beyond_end(self, cache):
        cache.append_audio(np.zeros(100, dtype=np.float32))
        audio = cache.get_audio(start_sample=200)
        assert len(audio) == 0

    def test_append_empty_array(self, cache):
        cache.append_audio(np.array([], dtype=np.float32))
        assert cache.total_samples == 0

    def test_append_rejects_2d(self, cache):
        with pytest.raises(ValueError, match="1-D"):
            cache.append_audio(np.zeros((2, 100), dtype=np.float32))

    def test_trim_front_basic(self, cache):
        cache.append_audio(np.arange(1000, dtype=np.float32))
        cache.trim_front(200)
        assert cache.total_samples == 800
        audio = cache.get_audio()
        np.testing.assert_array_equal(audio[:5], np.arange(200, 205, dtype=np.float32))

    def test_trim_front_all(self, cache):
        cache.append_audio(np.zeros(100, dtype=np.float32))
        cache.trim_front(100)
        assert cache.total_samples == 0
        assert len(cache.get_audio()) == 0

    def test_trim_front_more_than_available(self, cache):
        cache.append_audio(np.zeros(100, dtype=np.float32))
        cache.trim_front(500)
        assert cache.total_samples == 0

    def test_trim_front_zero(self, cache):
        cache.append_audio(np.zeros(100, dtype=np.float32))
        cache.trim_front(0)
        assert cache.total_samples == 100

    def test_trim_front_negative(self, cache):
        cache.append_audio(np.zeros(100, dtype=np.float32))
        cache.trim_front(-10)
        assert cache.total_samples == 100

    def test_reset(self, cache):
        cache.append_audio(np.zeros(1000, dtype=np.float32))
        cache.reset()
        assert cache.total_samples == 0
        assert cache.duration_sec == 0.0
        assert len(cache.get_audio()) == 0


# ── Mel feature computation tests ────────────────────────────────────


class TestMelFeatures:
    def test_compute_empty(self, cache):
        features, masks = cache.compute_features()
        assert features == []
        assert masks == []

    def test_compute_short_audio(self, cache, fe):
        """Audio shorter than one chunk (< 30s)."""
        cache.append_audio(np.random.randn(160_000).astype(np.float32))  # 10s
        features, masks = cache.compute_features()
        assert len(features) == 1
        assert features[0].shape == (128, 3000)
        assert masks[0].shape == (3000,)
        assert fe.call_count == 1

    def test_compute_full_chunk(self, cache, fe):
        """Exactly 30s of audio → 1 chunk."""
        cache.append_audio(np.random.randn(480_000).astype(np.float32))
        features, masks = cache.compute_features()
        assert len(features) == 1

    def test_compute_multiple_chunks(self, cache, fe):
        """35s of audio → 2 chunks."""
        cache.append_audio(np.random.randn(560_000).astype(np.float32))
        features, masks = cache.compute_features()
        assert len(features) == 2
        assert fe.call_count == 1
        # Verify chunks were split correctly
        assert len(fe.last_chunks) == 2
        assert len(fe.last_chunks[0]) == 480_000
        assert len(fe.last_chunks[1]) == 80_000

    def test_compute_with_offset(self, cache, fe):
        """Compute features starting from an offset."""
        cache.append_audio(np.random.randn(320_000).astype(np.float32))  # 20s
        features, masks = cache.compute_features(start_sample=160_000)  # skip 10s
        assert len(features) == 1
        assert len(fe.last_chunks[0]) == 160_000  # only 10s of audio

    def test_compute_offset_beyond_end(self, cache):
        cache.append_audio(np.random.randn(100).astype(np.float32))
        features, masks = cache.compute_features(start_sample=200)
        assert features == []
        assert masks == []

    def test_get_num_chunks(self, cache):
        assert cache.get_num_chunks() == 0

        cache.append_audio(np.zeros(160_000, dtype=np.float32))  # 10s
        assert cache.get_num_chunks() == 1

        cache.append_audio(np.zeros(320_000, dtype=np.float32))  # +20s = 30s
        assert cache.get_num_chunks() == 1

        cache.append_audio(np.zeros(1, dtype=np.float32))  # 30s + 1 sample
        assert cache.get_num_chunks() == 2

    def test_get_num_chunks_with_offset(self, cache):
        cache.append_audio(np.zeros(560_000, dtype=np.float32))  # 35s
        assert cache.get_num_chunks(start_sample=0) == 2
        assert cache.get_num_chunks(start_sample=480_000) == 1
        assert cache.get_num_chunks(start_sample=560_000) == 0

    def test_feature_extractor_receives_correct_audio(self, cache, fe):
        """Verify the feature extractor receives the right audio data."""
        audio = np.arange(100, dtype=np.float32)
        cache.append_audio(audio)
        cache.compute_features()
        np.testing.assert_array_equal(fe.last_chunks[0], audio)


# ── Chunk splitting tests ────────────────────────────────────────────


class TestChunkSplitting:
    def test_split_single_chunk(self, cache):
        audio = np.zeros(100, dtype=np.float32)
        chunks = cache._split_into_chunks(audio)
        assert len(chunks) == 1
        assert len(chunks[0]) == 100

    def test_split_exactly_at_boundary(self, cache):
        audio = np.zeros(480_000, dtype=np.float32)  # exactly 30s
        chunks = cache._split_into_chunks(audio)
        assert len(chunks) == 1

    def test_split_just_over_boundary(self, cache):
        audio = np.zeros(480_001, dtype=np.float32)  # 30s + 1
        chunks = cache._split_into_chunks(audio)
        assert len(chunks) == 2
        assert len(chunks[0]) == 480_000
        assert len(chunks[1]) == 1

    def test_split_preserves_data(self, cache):
        audio = np.arange(960_000, dtype=np.float32)  # 60s → 2 chunks
        chunks = cache._split_into_chunks(audio)
        recombined = np.concatenate(chunks)
        np.testing.assert_array_equal(recombined, audio)


# ── Integration-style tests ──────────────────────────────────────────


class TestStreamingWorkflow:
    """Simulate the streaming ASR workflow with the mel cache."""

    def test_incremental_append_and_compute(self, fe):
        """Simulate 10 steps of 0.5s audio appends."""
        cache = MelCache(fe, whisper_chunk_size=30, sample_rate=16_000)

        for step in range(10):
            chunk = np.random.randn(8000).astype(np.float32)  # 0.5s
            cache.append_audio(chunk)
            features, masks = cache.compute_features()
            assert len(features) == 1  # always 1 chunk (max 5s < 30s)

        assert cache.total_samples == 80_000  # 5s total
        assert fe.call_count == 10

    def test_trim_and_recompute(self, fe):
        """Simulate buffer trimming after exceeding max_audio."""
        cache = MelCache(fe, whisper_chunk_size=30, sample_rate=16_000)

        # Add 22s of audio
        for _ in range(44):
            cache.append_audio(np.zeros(8000, dtype=np.float32))

        assert abs(cache.duration_sec - 22.0) < 1e-6

        # Trim 4s from front
        cache.trim_front(64_000)
        assert abs(cache.duration_sec - 18.0) < 1e-6

        # Features should still work
        features, masks = cache.compute_features()
        assert len(features) == 1

    def test_do_normalize_passthrough(self, fe):
        """Verify do_normalize is passed to the feature extractor."""

        class NormalizeTracker(MockFeatureExtractor):
            def __call__(self, *args, **kwargs):
                self.last_do_normalize = kwargs.get("do_normalize")
                return super().__call__(*args, **kwargs)

        tracker = NormalizeTracker()
        cache = MelCache(tracker, do_normalize=False)
        cache.append_audio(np.zeros(8000, dtype=np.float32))
        cache.compute_features()
        assert tracker.last_do_normalize is False

        cache2 = MelCache(tracker, do_normalize=True)
        cache2.append_audio(np.zeros(8000, dtype=np.float32))
        cache2.compute_features()
        assert tracker.last_do_normalize is True


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_very_short_audio(self, cache):
        """Single sample."""
        cache.append_audio(np.array([0.5], dtype=np.float32))
        assert cache.total_samples == 1
        features, masks = cache.compute_features()
        assert len(features) == 1

    def test_type_conversion(self, cache):
        """Float64 input should be converted to float32."""
        chunk = np.zeros(100, dtype=np.float64)
        cache.append_audio(chunk)
        audio = cache.get_audio()
        assert audio.dtype == np.float32

    def test_repeated_get_audio_is_consistent(self, cache):
        """Multiple get_audio calls should return the same data."""
        cache.append_audio(np.arange(100, dtype=np.float32))
        a1 = cache.get_audio()
        a2 = cache.get_audio()
        np.testing.assert_array_equal(a1, a2)

    def test_append_after_trim(self, cache):
        """Append new data after trimming."""
        cache.append_audio(np.ones(100, dtype=np.float32))
        cache.trim_front(50)
        cache.append_audio(np.ones(100, dtype=np.float32) * 2.0)
        audio = cache.get_audio()
        assert len(audio) == 150
        np.testing.assert_array_equal(audio[:50], 1.0)
        np.testing.assert_array_equal(audio[50:], 2.0)

    def test_zero_sample_rate(self):
        """Zero sample rate should not crash duration_sec."""
        fe = MockFeatureExtractor()
        cache = MelCache(fe, sample_rate=0)
        cache._total_samples = 100
        assert cache.duration_sec == 0.0
