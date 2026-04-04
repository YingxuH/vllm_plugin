"""Per-connection streaming ASR session state.

Manages the sliding audio buffer, prefix continuation, step counting
(display vs append), and buffer trimming.  This module has no vLLM
dependency and can be tested independently.
"""

from __future__ import annotations

import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SessionConfig:
    """Immutable configuration for a streaming session."""

    display_delta: float = 0.5
    append_delta: float = 4.0
    max_audio: float = 20.0
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        if self.display_delta <= 0:
            raise ValueError(f"display_delta must be > 0, got {self.display_delta}")
        if self.append_delta <= 0:
            raise ValueError(f"append_delta must be > 0, got {self.append_delta}")
        if self.max_audio <= 0:
            raise ValueError(f"max_audio must be > 0, got {self.max_audio}")
        if self.append_delta < self.display_delta:
            raise ValueError(
                f"append_delta ({self.append_delta}) must be >= "
                f"display_delta ({self.display_delta})"
            )
        ratio = self.append_delta / self.display_delta
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"append_delta ({self.append_delta}) must be an integer "
                f"multiple of display_delta ({self.display_delta})"
            )

    @property
    def steps_per_append(self) -> int:
        return max(1, round(self.append_delta / self.display_delta))


_SPEAKER_TAG_RE = re.compile(r"<speaker\s*\d+\s*>\s*:?", flags=re.IGNORECASE)
_SPEAKER_WORD_RE = re.compile(r"\bSpeaker\s*\d+\s*:?", flags=re.IGNORECASE)


def strip_speaker_tags(text: str) -> str:
    """Strip AudioLLM speaker tags from text."""
    text = _SPEAKER_TAG_RE.sub("", text)
    text = _SPEAKER_WORD_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class StreamingASRSession:
    """Per-connection streaming ASR state.

    Manages audio buffering, prefix continuation for the AudioLLM,
    step counting, and buffer trimming.

    Parameters
    ----------
    config:
        Session configuration (deltas, max_audio, sample_rate).
    session_id:
        Unique session identifier (auto-generated if omitted).
    """

    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.config = config or SessionConfig()
        self.session_id = session_id or str(uuid.uuid4())

        # Audio buffer
        self._chunks: list[np.ndarray] = []
        self._total_samples: int = 0
        self._flat: Optional[np.ndarray] = None

        # Prefix state (AudioLLM stream)
        self.prefix_raw: str = ""
        self.committed_text: str = ""
        self.audio_start: float = 0.0  # left edge of buffer (seconds)
        self.prefix_checkpoints: deque[int] = deque()

        # Step counter
        self._step_count: int = 0

        # Timestamps
        self.created_at: float = time.time()
        self.last_active: float = self.created_at

    # ── Audio buffer ─────────────────────────────────────────────────

    def append_audio(self, chunk: np.ndarray) -> None:
        """Append a chunk of float32 mono audio."""
        if chunk.ndim != 1:
            raise ValueError(f"Expected 1-D audio, got shape {chunk.shape}")
        if len(chunk) == 0:
            return
        self._chunks.append(chunk.astype(np.float32, copy=False))
        self._total_samples += len(chunk)
        self._flat = None
        self.last_active = time.time()

    @property
    def audio_buffer(self) -> np.ndarray:
        """Full contiguous audio buffer."""
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

    @property
    def total_samples(self) -> int:
        return self._total_samples

    @property
    def buffer_duration_sec(self) -> float:
        """Duration of the active audio window (from audio_start to end)."""
        total_dur = self._total_samples / self.config.sample_rate
        return max(0.0, total_dur - self.audio_start)

    def get_active_audio(self) -> np.ndarray:
        """Return audio from audio_start to end (the active window)."""
        start_sample = int(self.audio_start * self.config.sample_rate)
        buf = self.audio_buffer
        return buf[start_sample:] if start_sample < len(buf) else np.array([], dtype=np.float32)

    # ── Step management ──────────────────────────────────────────────

    @property
    def step_count(self) -> int:
        return self._step_count

    def advance_step(self) -> dict:
        """Advance the step counter and return step metadata.

        Returns a dict with ``step``, ``is_append``, ``cycle_pos``.
        """
        self._step_count += 1
        self.last_active = time.time()
        spc = self.config.steps_per_append
        is_append = (self._step_count % spc) == 0
        cycle_pos = self._step_count % spc  # 0 = append
        return {
            "step": self._step_count,
            "is_append": is_append,
            "cycle_pos": cycle_pos,
        }

    @property
    def is_append_step(self) -> bool:
        """Whether the *current* step is an append step."""
        if self._step_count == 0:
            return False
        return (self._step_count % self.config.steps_per_append) == 0

    # ── Prefix management ────────────────────────────────────────────

    def update_prefix(self, raw_text: str) -> None:
        """Commit LLM output to prefix (call only at append steps).

        Appends *raw_text* to ``prefix_raw`` and records a checkpoint
        for future buffer trimming.
        """
        if self.prefix_raw:
            self.prefix_raw = self.prefix_raw.rstrip() + raw_text
        else:
            self.prefix_raw = raw_text
        self.prefix_checkpoints.append(len(self.prefix_raw))

    def build_display_text(self, raw_text: str, is_append: bool) -> str:
        """Build the full user-visible text for a step.

        At append steps, ``raw_text`` is already committed to prefix.
        At display-only steps, combines prefix + raw_text.
        """
        if is_append:
            display_raw = self.prefix_raw
        else:
            if self.prefix_raw:
                display_raw = self.prefix_raw.rstrip() + raw_text
            else:
                display_raw = raw_text

        display_clean = strip_speaker_tags(display_raw).strip()
        if self.committed_text:
            return self.committed_text + " " + display_clean
        return display_clean

    def get_final_text(self) -> str:
        """Build the final transcript from committed + remaining prefix."""
        remaining = strip_speaker_tags(self.prefix_raw).strip()
        if remaining:
            if self.committed_text:
                return self.committed_text + " " + remaining
            return remaining
        return self.committed_text

    # ── Buffer trimming ──────────────────────────────────────────────

    def trim_if_needed(self) -> None:
        """Trim oldest audio when buffer exceeds ``max_audio``.

        Pops prefix checkpoints and advances ``audio_start`` by
        ``append_delta`` for each trimmed segment.
        """
        while self.buffer_duration_sec > self.config.max_audio + 1e-9:
            if not self.prefix_checkpoints:
                break
            trim_at = self.prefix_checkpoints.popleft()
            committed_raw = self.prefix_raw[:trim_at]
            committed_clean = strip_speaker_tags(committed_raw).strip()
            if committed_clean:
                self.committed_text += (
                    (" " if self.committed_text else "") + committed_clean
                )
            self.prefix_raw = self.prefix_raw[trim_at:]
            # Adjust remaining checkpoints
            for i in range(len(self.prefix_checkpoints)):
                self.prefix_checkpoints[i] -= trim_at
            self.audio_start += self.config.append_delta

    # ── Lifecycle ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all session state."""
        self._chunks.clear()
        self._total_samples = 0
        self._flat = None
        self.prefix_raw = ""
        self.committed_text = ""
        self.audio_start = 0.0
        self.prefix_checkpoints.clear()
        self._step_count = 0
