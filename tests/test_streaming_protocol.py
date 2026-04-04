"""Unit tests for protocol.py — WebSocket message types and helpers."""

import base64

import numpy as np
import pytest

from vllm_plugin_meralion2.streaming.protocol import (
    AudioChunkMessage,
    ErrorResponse,
    FinalResponse,
    SessionEndMessage,
    SessionStartMessage,
    SessionStartedResponse,
    TranscriptResponse,
    decode_pcm16,
    parse_client_message,
)


# ── SessionStartMessage tests ────────────────────────────────────────


class TestSessionStartMessage:
    def test_defaults(self):
        msg = SessionStartMessage()
        assert msg.type == "session.start"
        assert msg.display_delta == 0.5
        assert msg.append_delta == 4.0
        assert msg.max_audio == 20.0

    def test_custom_values(self):
        msg = SessionStartMessage(display_delta=1.0, append_delta=5.0, max_audio=30.0)
        assert msg.display_delta == 1.0
        assert msg.append_delta == 5.0
        assert msg.max_audio == 30.0

    def test_negative_display_delta(self):
        with pytest.raises(Exception):
            SessionStartMessage(display_delta=-1.0)

    def test_zero_display_delta(self):
        with pytest.raises(Exception):
            SessionStartMessage(display_delta=0)

    def test_negative_append_delta(self):
        with pytest.raises(Exception):
            SessionStartMessage(append_delta=-1.0)

    def test_negative_max_audio(self):
        with pytest.raises(Exception):
            SessionStartMessage(max_audio=-10.0)

    def test_serialization(self):
        msg = SessionStartMessage()
        d = msg.model_dump()
        assert d["type"] == "session.start"
        assert d["display_delta"] == 0.5


# ── AudioChunkMessage tests ──────────────────────────────────────────


class TestAudioChunkMessage:
    def test_basic(self):
        msg = AudioChunkMessage(data="AQID")
        assert msg.type == "audio_chunk"
        assert msg.data == "AQID"

    def test_serialization(self):
        msg = AudioChunkMessage(data="abc123")
        d = msg.model_dump()
        assert d["type"] == "audio_chunk"
        assert d["data"] == "abc123"


# ── SessionEndMessage tests ──────────────────────────────────────────


class TestSessionEndMessage:
    def test_type(self):
        msg = SessionEndMessage()
        assert msg.type == "session.end"


# ── Server response tests ────────────────────────────────────────────


class TestSessionStartedResponse:
    def test_fields(self):
        resp = SessionStartedResponse(session_id="abc-123")
        assert resp.type == "session.started"
        assert resp.session_id == "abc-123"

    def test_serialization(self):
        resp = SessionStartedResponse(session_id="x")
        d = resp.model_dump()
        assert d["type"] == "session.started"
        assert d["session_id"] == "x"


class TestTranscriptResponse:
    def test_fields(self):
        resp = TranscriptResponse(
            text="hello world",
            raw="<Speaker1>: hello world",
            is_append=True,
            step=8,
            comp_tokens=15,
        )
        assert resp.type == "transcript"
        assert resp.text == "hello world"
        assert resp.is_append is True
        assert resp.step == 8
        assert resp.comp_tokens == 15

    def test_default_comp_tokens(self):
        resp = TranscriptResponse(text="", raw="", is_append=False, step=1)
        assert resp.comp_tokens == 0


class TestFinalResponse:
    def test_fields(self):
        resp = FinalResponse(text="full transcript")
        assert resp.type == "final"
        assert resp.text == "full transcript"


class TestErrorResponse:
    def test_fields(self):
        resp = ErrorResponse(message="bad audio", code="invalid_audio")
        assert resp.type == "error"
        assert resp.message == "bad audio"
        assert resp.code == "invalid_audio"

    def test_optional_code(self):
        resp = ErrorResponse(message="oops")
        assert resp.code is None

    def test_serialization(self):
        resp = ErrorResponse(message="test", code="test_code")
        d = resp.model_dump()
        assert d["type"] == "error"
        assert d["message"] == "test"
        assert d["code"] == "test_code"


# ── decode_pcm16 tests ──────────────────────────────────────────────


class TestDecodePcm16:
    def _encode_pcm16(self, samples: np.ndarray) -> str:
        """Helper: encode float32 audio to base64 PCM16."""
        pcm = np.clip(samples, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)
        return base64.b64encode(pcm.tobytes()).decode()

    def test_basic_decode(self):
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        b64 = self._encode_pcm16(original)
        decoded = decode_pcm16(b64)
        # Check approximate equality (quantization error)
        np.testing.assert_allclose(decoded, original, atol=1.0 / 32767)

    def test_decode_zeros(self):
        b64 = self._encode_pcm16(np.zeros(100, dtype=np.float32))
        decoded = decode_pcm16(b64)
        assert len(decoded) == 100
        np.testing.assert_array_equal(decoded, 0.0)

    def test_output_range(self):
        original = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        b64 = self._encode_pcm16(original)
        decoded = decode_pcm16(b64)
        assert decoded.min() >= -1.0
        assert decoded.max() <= 1.0

    def test_output_dtype(self):
        b64 = self._encode_pcm16(np.zeros(10, dtype=np.float32))
        decoded = decode_pcm16(b64)
        assert decoded.dtype == np.float32

    def test_empty_data(self):
        with pytest.raises(ValueError, match="Empty audio"):
            decode_pcm16(base64.b64encode(b"").decode())

    def test_odd_byte_count(self):
        # 3 bytes can't form valid PCM16 samples
        b64 = base64.b64encode(b"\x00\x01\x02").decode()
        with pytest.raises(ValueError, match="even byte count"):
            decode_pcm16(b64)

    def test_invalid_base64(self):
        with pytest.raises(ValueError, match="Invalid base64"):
            decode_pcm16("not!valid!base64!!!")

    def test_single_sample(self):
        b64 = self._encode_pcm16(np.array([0.42], dtype=np.float32))
        decoded = decode_pcm16(b64)
        assert len(decoded) == 1
        np.testing.assert_allclose(decoded[0], 0.42, atol=1.0 / 32767)

    def test_large_buffer(self):
        """16000 samples = 1 second at 16 kHz."""
        # Use uniform [-0.9, 0.9] to stay within clipping range
        np.random.seed(42)
        original = (np.random.rand(16000).astype(np.float32) * 1.8 - 0.9)
        b64 = self._encode_pcm16(original)
        decoded = decode_pcm16(b64)
        assert len(decoded) == 16000
        np.testing.assert_allclose(decoded, original, atol=1.0 / 32767)


# ── parse_client_message tests ───────────────────────────────────────


class TestParseClientMessage:
    def test_session_start(self):
        msg = parse_client_message({"type": "session.start"})
        assert isinstance(msg, SessionStartMessage)

    def test_session_start_with_config(self):
        msg = parse_client_message({
            "type": "session.start",
            "display_delta": 1.0,
            "append_delta": 3.0,
            "max_audio": 15.0,
        })
        assert isinstance(msg, SessionStartMessage)
        assert msg.display_delta == 1.0
        assert msg.append_delta == 3.0
        assert msg.max_audio == 15.0

    def test_audio_chunk(self):
        msg = parse_client_message({"type": "audio_chunk", "data": "abc"})
        assert isinstance(msg, AudioChunkMessage)
        assert msg.data == "abc"

    def test_session_end(self):
        msg = parse_client_message({"type": "session.end"})
        assert isinstance(msg, SessionEndMessage)

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown message type"):
            parse_client_message({"type": "foobar"})

    def test_missing_type(self):
        with pytest.raises(ValueError, match="Unknown message type"):
            parse_client_message({})

    def test_audio_chunk_missing_data(self):
        with pytest.raises(Exception):
            parse_client_message({"type": "audio_chunk"})

    def test_session_start_invalid_delta(self):
        with pytest.raises(Exception):
            parse_client_message({"type": "session.start", "display_delta": -1.0})
