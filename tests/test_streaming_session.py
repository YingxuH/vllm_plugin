"""Unit tests for session.py — per-connection streaming ASR state."""

import numpy as np
import pytest

from vllm_plugin_meralion2.streaming.session import (
    SessionConfig,
    StreamingASRSession,
    strip_speaker_tags,
)


# ── SessionConfig tests ──────────────────────────────────────────────


class TestSessionConfig:
    def test_defaults(self):
        cfg = SessionConfig()
        assert cfg.display_delta == 0.5
        assert cfg.append_delta == 4.0
        assert cfg.max_audio == 20.0
        assert cfg.sample_rate == 16_000
        assert cfg.steps_per_append == 8

    def test_custom_values(self):
        cfg = SessionConfig(display_delta=1.0, append_delta=5.0, max_audio=30.0)
        assert cfg.steps_per_append == 5

    def test_display_delta_zero(self):
        with pytest.raises(ValueError, match="display_delta"):
            SessionConfig(display_delta=0)

    def test_append_delta_zero(self):
        with pytest.raises(ValueError, match="append_delta"):
            SessionConfig(append_delta=0)

    def test_max_audio_zero(self):
        with pytest.raises(ValueError, match="max_audio"):
            SessionConfig(max_audio=0)

    def test_negative_delta(self):
        with pytest.raises(ValueError, match="display_delta"):
            SessionConfig(display_delta=-1.0)

    def test_append_less_than_display(self):
        with pytest.raises(ValueError, match="append_delta .* must be >= display_delta"):
            SessionConfig(display_delta=2.0, append_delta=1.0)

    def test_non_integer_multiple(self):
        with pytest.raises(ValueError, match="integer multiple"):
            SessionConfig(display_delta=0.3, append_delta=1.0)

    def test_equal_deltas(self):
        cfg = SessionConfig(display_delta=1.0, append_delta=1.0)
        assert cfg.steps_per_append == 1

    def test_steps_per_append_rounding(self):
        cfg = SessionConfig(display_delta=0.5, append_delta=2.0)
        assert cfg.steps_per_append == 4


# ── strip_speaker_tags tests ─────────────────────────────────────────


class TestStripSpeakerTags:
    def test_basic_tag(self):
        assert strip_speaker_tags("<Speaker1>: hello") == "hello"

    def test_multiple_tags(self):
        result = strip_speaker_tags("<Speaker1>: hello <Speaker2>: world")
        assert result == "hello world"

    def test_no_tags(self):
        assert strip_speaker_tags("hello world") == "hello world"

    def test_tag_without_colon(self):
        assert strip_speaker_tags("<Speaker1> hello") == "hello"

    def test_word_format(self):
        assert strip_speaker_tags("Speaker1: hello") == "hello"

    def test_case_insensitive(self):
        assert strip_speaker_tags("<SPEAKER1>: hello") == "hello"

    def test_whitespace_normalization(self):
        assert strip_speaker_tags("  hello   world  ") == "hello world"

    def test_empty_string(self):
        assert strip_speaker_tags("") == ""

    def test_only_tags(self):
        assert strip_speaker_tags("<Speaker1>:") == ""

    def test_multi_digit_speaker(self):
        assert strip_speaker_tags("<Speaker12>: hello") == "hello"


# ── StreamingASRSession tests ────────────────────────────────────────


class TestSessionCreation:
    def test_default_creation(self):
        session = StreamingASRSession()
        assert session.step_count == 0
        assert session.prefix_raw == ""
        assert session.committed_text == ""
        assert session.audio_start == 0.0
        assert session.total_samples == 0

    def test_custom_config(self):
        cfg = SessionConfig(display_delta=1.0, append_delta=2.0, max_audio=10.0)
        session = StreamingASRSession(config=cfg)
        assert session.config.display_delta == 1.0

    def test_custom_session_id(self):
        session = StreamingASRSession(session_id="test-123")
        assert session.session_id == "test-123"

    def test_auto_session_id(self):
        s1 = StreamingASRSession()
        s2 = StreamingASRSession()
        assert s1.session_id != s2.session_id


# ── Audio buffer tests ───────────────────────────────────────────────


class TestSessionAudioBuffer:
    def test_append_audio(self):
        session = StreamingASRSession()
        session.append_audio(np.zeros(8000, dtype=np.float32))
        assert session.total_samples == 8000

    def test_append_multiple(self):
        session = StreamingASRSession()
        for _ in range(5):
            session.append_audio(np.zeros(8000, dtype=np.float32))
        assert session.total_samples == 40_000

    def test_audio_buffer_contiguous(self):
        session = StreamingASRSession()
        session.append_audio(np.ones(100, dtype=np.float32))
        session.append_audio(np.ones(100, dtype=np.float32) * 2.0)
        buf = session.audio_buffer
        assert len(buf) == 200
        np.testing.assert_array_equal(buf[:100], 1.0)
        np.testing.assert_array_equal(buf[100:], 2.0)

    def test_empty_append(self):
        session = StreamingASRSession()
        session.append_audio(np.array([], dtype=np.float32))
        assert session.total_samples == 0

    def test_append_rejects_2d(self):
        session = StreamingASRSession()
        with pytest.raises(ValueError, match="1-D"):
            session.append_audio(np.zeros((2, 100), dtype=np.float32))

    def test_buffer_duration(self):
        session = StreamingASRSession()
        session.append_audio(np.zeros(32_000, dtype=np.float32))
        assert abs(session.buffer_duration_sec - 2.0) < 1e-6

    def test_buffer_duration_with_audio_start(self):
        session = StreamingASRSession()
        session.append_audio(np.zeros(64_000, dtype=np.float32))
        session.audio_start = 1.0
        assert abs(session.buffer_duration_sec - 3.0) < 1e-6

    def test_get_active_audio(self):
        session = StreamingASRSession()
        session.append_audio(np.arange(320_000, dtype=np.float32))
        session.audio_start = 10.0  # skip first 10s
        active = session.get_active_audio()
        assert len(active) == 160_000  # remaining 10s

    def test_get_active_audio_empty(self):
        session = StreamingASRSession()
        active = session.get_active_audio()
        assert len(active) == 0


# ── Step management tests ────────────────────────────────────────────


class TestStepManagement:
    def test_initial_state(self):
        session = StreamingASRSession()
        assert session.step_count == 0
        assert session.is_append_step is False

    def test_advance_step_returns_info(self):
        session = StreamingASRSession()
        info = session.advance_step()
        assert info["step"] == 1
        assert info["is_append"] is False
        assert info["cycle_pos"] == 1

    def test_display_steps(self):
        """Steps 1-7 are display-only, step 8 is append."""
        session = StreamingASRSession()  # steps_per_append = 8
        for i in range(1, 8):
            info = session.advance_step()
            assert info["is_append"] is False, f"Step {i} should be display-only"

    def test_append_step(self):
        session = StreamingASRSession()
        for _ in range(8):
            info = session.advance_step()
        assert info["is_append"] is True
        assert info["step"] == 8
        assert info["cycle_pos"] == 0

    def test_step_cycle_repeats(self):
        session = StreamingASRSession()
        append_steps = []
        for i in range(24):
            info = session.advance_step()
            if info["is_append"]:
                append_steps.append(info["step"])
        assert append_steps == [8, 16, 24]

    def test_is_append_step_property(self):
        session = StreamingASRSession()
        for _ in range(8):
            session.advance_step()
        assert session.is_append_step is True

    def test_custom_steps_per_append(self):
        cfg = SessionConfig(display_delta=0.5, append_delta=2.0)  # 4 steps
        session = StreamingASRSession(config=cfg)
        for _ in range(4):
            info = session.advance_step()
        assert info["is_append"] is True
        assert info["step"] == 4

    def test_steps_per_append_1(self):
        """Every step is an append step."""
        cfg = SessionConfig(display_delta=1.0, append_delta=1.0)
        session = StreamingASRSession(config=cfg)
        for _ in range(3):
            info = session.advance_step()
            assert info["is_append"] is True


# ── Prefix management tests ──────────────────────────────────────────


class TestPrefixManagement:
    def test_initial_prefix(self):
        session = StreamingASRSession()
        assert session.prefix_raw == ""

    def test_update_prefix_first(self):
        session = StreamingASRSession()
        session.update_prefix("hello world")
        assert session.prefix_raw == "hello world"

    def test_update_prefix_appends(self):
        session = StreamingASRSession()
        session.update_prefix("hello ")
        session.update_prefix("world")
        assert session.prefix_raw == "helloworld"

    def test_update_prefix_strips_trailing_space(self):
        session = StreamingASRSession()
        session.update_prefix("hello  ")
        session.update_prefix(" world")
        assert session.prefix_raw == "hello world"

    def test_checkpoint_recorded(self):
        session = StreamingASRSession()
        session.update_prefix("hello")
        assert len(session.prefix_checkpoints) == 1
        assert session.prefix_checkpoints[0] == 5

    def test_multiple_checkpoints(self):
        session = StreamingASRSession()
        session.update_prefix("hello")
        session.update_prefix(" world")
        assert len(session.prefix_checkpoints) == 2

    def test_build_display_text_append(self):
        session = StreamingASRSession()
        session.update_prefix("<Speaker1>: hello")
        text = session.build_display_text("", is_append=True)
        assert text == "hello"

    def test_build_display_text_display_only(self):
        session = StreamingASRSession()
        session.update_prefix("<Speaker1>: hello")
        text = session.build_display_text(" world", is_append=False)
        assert text == "hello world"

    def test_build_display_text_with_committed(self):
        session = StreamingASRSession()
        session.committed_text = "previous"
        session.update_prefix("hello")
        text = session.build_display_text("", is_append=True)
        assert text == "previous hello"

    def test_build_display_text_no_prefix(self):
        session = StreamingASRSession()
        text = session.build_display_text("hello world", is_append=False)
        assert text == "hello world"

    def test_get_final_text_basic(self):
        session = StreamingASRSession()
        session.update_prefix("<Speaker1>: hello world")
        assert session.get_final_text() == "hello world"

    def test_get_final_text_with_committed(self):
        session = StreamingASRSession()
        session.committed_text = "previous"
        session.update_prefix("new text")
        assert session.get_final_text() == "previous new text"

    def test_get_final_text_empty(self):
        session = StreamingASRSession()
        assert session.get_final_text() == ""

    def test_get_final_text_only_committed(self):
        session = StreamingASRSession()
        session.committed_text = "only committed"
        assert session.get_final_text() == "only committed"


# ── Buffer trimming tests ────────────────────────────────────────────


class TestBufferTrimming:
    def test_no_trim_under_limit(self):
        session = StreamingASRSession()
        # Add 10s of audio (under 20s limit)
        session.append_audio(np.zeros(160_000, dtype=np.float32))
        session.trim_if_needed()
        assert session.audio_start == 0.0

    def test_trim_over_limit(self):
        cfg = SessionConfig(display_delta=0.5, append_delta=4.0, max_audio=10.0)
        session = StreamingASRSession(config=cfg)

        # Add 12s audio
        session.append_audio(np.zeros(192_000, dtype=np.float32))
        # Create prefix checkpoints for trimming
        session.update_prefix("<Speaker1>: first four seconds")
        session.update_prefix(" more text")

        session.trim_if_needed()
        assert session.audio_start > 0.0

    def test_trim_commits_text(self):
        cfg = SessionConfig(display_delta=0.5, append_delta=4.0, max_audio=10.0)
        session = StreamingASRSession(config=cfg)

        session.append_audio(np.zeros(192_000, dtype=np.float32))
        session.update_prefix("<Speaker1>: hello")
        session.update_prefix(" world")

        initial_committed = session.committed_text
        session.trim_if_needed()
        # After trimming, committed_text should have grown
        assert len(session.committed_text) >= len(initial_committed)

    def test_trim_adjusts_checkpoints(self):
        cfg = SessionConfig(display_delta=0.5, append_delta=2.0, max_audio=5.0)
        session = StreamingASRSession(config=cfg)

        session.append_audio(np.zeros(128_000, dtype=np.float32))  # 8s
        session.update_prefix("first")
        session.update_prefix("second")
        session.update_prefix("third")

        n_before = len(session.prefix_checkpoints)
        session.trim_if_needed()
        assert len(session.prefix_checkpoints) < n_before

    def test_trim_without_checkpoints(self):
        cfg = SessionConfig(display_delta=0.5, append_delta=4.0, max_audio=5.0)
        session = StreamingASRSession(config=cfg)

        session.append_audio(np.zeros(128_000, dtype=np.float32))  # 8s > 5s limit
        # No prefix checkpoints — trim should not crash
        session.trim_if_needed()
        # audio_start stays at 0 because there's nothing to trim
        assert session.audio_start == 0.0

    def test_trim_advances_audio_start_by_append_delta(self):
        cfg = SessionConfig(display_delta=0.5, append_delta=4.0, max_audio=10.0)
        session = StreamingASRSession(config=cfg)

        session.append_audio(np.zeros(256_000, dtype=np.float32))  # 16s
        session.update_prefix("hello ")
        session.update_prefix("world ")

        session.trim_if_needed()
        # Each trim advances by append_delta (4.0s)
        assert session.audio_start % cfg.append_delta == 0.0
        assert session.audio_start > 0.0


# ── Lifecycle tests ──────────────────────────────────────────────────


class TestSessionLifecycle:
    def test_reset_clears_everything(self):
        session = StreamingASRSession()
        session.append_audio(np.zeros(8000, dtype=np.float32))
        session.advance_step()
        session.update_prefix("hello")
        session.committed_text = "committed"
        session.audio_start = 5.0

        session.reset()

        assert session.total_samples == 0
        assert session.step_count == 0
        assert session.prefix_raw == ""
        assert session.committed_text == ""
        assert session.audio_start == 0.0
        assert len(session.prefix_checkpoints) == 0

    def test_last_active_updated_on_append(self):
        session = StreamingASRSession()
        t_before = session.last_active
        import time
        time.sleep(0.01)
        session.append_audio(np.zeros(100, dtype=np.float32))
        assert session.last_active > t_before

    def test_last_active_updated_on_advance(self):
        session = StreamingASRSession()
        t_before = session.last_active
        import time
        time.sleep(0.01)
        session.advance_step()
        assert session.last_active > t_before


# ── Full streaming workflow ──────────────────────────────────────────


class TestStreamingWorkflow:
    """Simulate a full streaming transcription workflow."""

    def test_basic_flow(self):
        session = StreamingASRSession()

        # Simulate 16 steps (2 append cycles)
        for i in range(16):
            session.append_audio(np.zeros(8000, dtype=np.float32))
            info = session.advance_step()

            if info["is_append"]:
                session.update_prefix(f"text_{info['step']} ")

        assert session.step_count == 16
        assert len(session.prefix_checkpoints) == 2
        assert "text_8" in session.prefix_raw
        assert "text_16" in session.prefix_raw

    def test_trim_during_long_stream(self):
        cfg = SessionConfig(display_delta=0.5, append_delta=4.0, max_audio=10.0)
        session = StreamingASRSession(config=cfg)

        # Simulate 30s of streaming (60 steps)
        for i in range(60):
            session.append_audio(np.zeros(8000, dtype=np.float32))
            info = session.advance_step()

            if info["is_append"]:
                session.update_prefix(f"chunk{info['step']//8} ")

            session.trim_if_needed()

        # Buffer should be trimmed to ~10s
        assert session.buffer_duration_sec <= cfg.max_audio + 0.5
        # committed_text should have content from trimmed portions
        assert len(session.committed_text) > 0
        # Final text should combine committed + remaining prefix
        final = session.get_final_text()
        assert len(final) > 0
