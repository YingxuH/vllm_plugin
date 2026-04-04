"""WebSocket protocol messages for streaming ASR.

Defines the message types exchanged between the client and the
streaming ASR WebSocket endpoint.

Client → Server
~~~~~~~~~~~~~~~~
- ``session.start``  — open a session with optional config overrides
- ``audio_chunk``    — base64-encoded PCM16 mono 16 kHz audio
- ``session.end``    — finalize and retrieve the final transcript

Server → Client
~~~~~~~~~~~~~~~~
- ``session.started`` — confirms session creation
- ``transcript``      — per-step transcription result
- ``final``           — final transcript after session.end
- ``error``           — error message
"""

from __future__ import annotations

import base64
from typing import Optional

import numpy as np
from pydantic import BaseModel, field_validator


# ── Client → Server ─────────────────────────────────────────────────


class SessionStartMessage(BaseModel):
    """Client request to start a streaming session."""

    type: str = "session.start"
    display_delta: float = 0.5
    append_delta: float = 4.0
    max_audio: float = 20.0

    @field_validator("display_delta", "append_delta", "max_audio")
    @classmethod
    def _positive(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"{info.field_name} must be > 0, got {v}")
        return v


class AudioChunkMessage(BaseModel):
    """Client sends a chunk of audio data."""

    type: str = "audio_chunk"
    data: str  # base64-encoded PCM16 @ 16 kHz mono


class SessionEndMessage(BaseModel):
    """Client signals end of audio stream."""

    type: str = "session.end"


# ── Server → Client ─────────────────────────────────────────────────


class SessionStartedResponse(BaseModel):
    """Confirms session creation."""

    type: str = "session.started"
    session_id: str


class TranscriptResponse(BaseModel):
    """Per-step transcription result."""

    type: str = "transcript"
    text: str
    raw: str
    is_append: bool
    step: int
    comp_tokens: int = 0


class FinalResponse(BaseModel):
    """Final transcript after session ends."""

    type: str = "final"
    text: str


class ErrorResponse(BaseModel):
    """Error message."""

    type: str = "error"
    message: str
    code: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────


def decode_pcm16(b64_data: str, sample_rate: int = 16_000) -> np.ndarray:
    """Decode base64-encoded PCM16 audio to float32 numpy array.

    Parameters
    ----------
    b64_data:
        Base64-encoded raw PCM16 bytes (little-endian, mono, 16 kHz).
    sample_rate:
        Expected sample rate (used only for validation context).

    Returns
    -------
    np.ndarray:
        Float32 audio array in [-1.0, 1.0].

    Raises
    ------
    ValueError:
        If the data cannot be decoded or has odd byte count.
    """
    try:
        pcm_bytes = base64.b64decode(b64_data)
    except Exception as exc:
        raise ValueError(f"Invalid base64 data: {exc}") from exc
    if len(pcm_bytes) == 0:
        raise ValueError("Empty audio data")
    if len(pcm_bytes) % 2 != 0:
        raise ValueError(
            f"PCM16 data must have even byte count, got {len(pcm_bytes)}"
        )
    pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm_int16.astype(np.float32) / 32767.0


def parse_client_message(raw: dict) -> SessionStartMessage | AudioChunkMessage | SessionEndMessage:
    """Parse a raw JSON dict into a typed client message.

    Raises
    ------
    ValueError:
        If the message type is unknown or fields are invalid.
    """
    msg_type = raw.get("type", "")
    if msg_type == "session.start":
        return SessionStartMessage(**raw)
    if msg_type == "audio_chunk":
        return AudioChunkMessage(**raw)
    if msg_type == "session.end":
        return SessionEndMessage(**raw)
    raise ValueError(f"Unknown message type: {msg_type!r}")
