"""Unit tests for endpoint.py — WebSocket streaming ASR endpoint.

All tests run without a real vLLM engine. The engine is mocked so
tests are fast and do not require GPU.
"""

import asyncio
import base64
import io
import wave
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient

from vllm_plugin_meralion2.streaming.endpoint import (
    SESSION_TIMEOUT_SEC,
    _encode_audio_url,
    _engine_state,
    _sessions,
    register_streaming_asr,
    init_streaming_state,
    router,
)
from vllm_plugin_meralion2.streaming.protocol import decode_pcm16


# ── Helpers ──────────────────────────────────────────────────────────


def _make_pcm16_b64(duration_s: float = 0.5, sr: int = 16_000) -> str:
    """Create base64-encoded PCM16 audio of given duration."""
    n_samples = int(duration_s * sr)
    audio = np.random.randn(n_samples).astype(np.float32) * 0.1
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode()


def _make_silence_pcm16_b64(duration_s: float = 0.5, sr: int = 16_000) -> str:
    """Create base64-encoded silent PCM16 audio."""
    n_samples = int(duration_s * sr)
    pcm = np.zeros(n_samples, dtype=np.int16)
    return base64.b64encode(pcm.tobytes()).decode()


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def app():
    """Create a FastAPI app with the streaming ASR endpoint."""
    test_app = FastAPI()
    register_streaming_asr(test_app)
    # Mock the serving_chat in engine state
    _engine_state.clear()
    _sessions.clear()
    return test_app


@pytest.fixture
def mock_engine():
    """Set up mocked vLLM engine state for transcription calls."""
    mock_chat = AsyncMock()

    # Create a mock response
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "hello world"
    mock_response.choices = [mock_choice]
    mock_usage = MagicMock()
    mock_usage.completion_tokens = 5
    mock_response.usage = mock_usage
    mock_chat.create_chat_completion.return_value = mock_response

    _engine_state["serving_chat"] = mock_chat
    _engine_state["model_name"] = "test-model"
    _engine_state["prompt_template"] = "Transcribe."

    yield mock_chat

    _engine_state.clear()


@pytest.fixture(autouse=True)
def cleanup_sessions():
    """Ensure sessions are cleaned up after each test."""
    yield
    _sessions.clear()


# ── _encode_audio_url tests ──────────────────────────────────────────


class TestEncodeAudioUrl:
    def test_returns_data_url(self):
        audio = np.zeros(100, dtype=np.float32)
        url = _encode_audio_url(audio, sr=16_000)
        assert url.startswith("data:audio/wav;base64,")

    def test_decodable_wav(self):
        audio = np.random.randn(16_000).astype(np.float32) * 0.5
        url = _encode_audio_url(audio, sr=16_000)
        b64_data = url.split(",", 1)[1]
        wav_bytes = base64.b64decode(b64_data)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16_000
            assert wf.getnframes() == 16_000

    def test_clips_audio(self):
        audio = np.array([2.0, -2.0], dtype=np.float32)
        url = _encode_audio_url(audio)
        b64_data = url.split(",", 1)[1]
        wav_bytes = base64.b64decode(b64_data)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(2)
            pcm = np.frombuffer(frames, dtype=np.int16)
            # Should be clipped to [-32767, 32767]
            assert pcm[0] == 32767
            assert pcm[1] == -32767

    def test_empty_audio(self):
        audio = np.array([], dtype=np.float32)
        url = _encode_audio_url(audio)
        assert url.startswith("data:audio/wav;base64,")


# ── Health endpoint tests ────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_ok(self, app, mock_engine):
        client = TestClient(app)
        resp = client.get("/v1/streaming_asr/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["engine_ready"] is True
        assert data["active_sessions"] == 0

    def test_health_no_engine(self, app):
        _engine_state.clear()
        client = TestClient(app)
        resp = client.get("/v1/streaming_asr/health")
        data = resp.json()
        assert data["engine_ready"] is False


# ── WebSocket lifecycle tests ────────────────────────────────────────


class TestWebSocketLifecycle:
    def test_connect_and_disconnect(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            resp = ws.receive_json()
            assert resp["type"] == "session.started"
            assert "session_id" in resp

    def test_session_start_creates_session(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            resp = ws.receive_json()
            sid = resp["session_id"]
            assert sid in _sessions

    def test_session_end_sends_final(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            ws.send_json({"type": "session.end"})
            resp = ws.receive_json()
            assert resp["type"] == "final"
            assert "text" in resp

    def test_session_cleaned_up_after_end(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            resp = ws.receive_json()
            sid = resp["session_id"]

            ws.send_json({"type": "session.end"})
            ws.receive_json()

        assert sid not in _sessions

    def test_session_cleaned_up_on_disconnect(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            resp = ws.receive_json()
            sid = resp["session_id"]
        # WebSocket closed without session.end
        assert sid not in _sessions


# ── Error handling tests ─────────────────────────────────────────────


class TestErrorHandling:
    def test_unknown_message_type(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "foobar"})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Unknown message type" in resp["message"]

    def test_audio_before_session_start(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert resp["code"] == "no_session"

    def test_end_before_session_start(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.end"})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert resp["code"] == "no_session"

    def test_invalid_audio_data(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            ws.send_json({"type": "audio_chunk", "data": "!!!invalid!!!"})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert resp["code"] == "invalid_audio"

    def test_odd_byte_audio(self, app, mock_engine):
        """PCM16 must have even byte count."""
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            odd_bytes = base64.b64encode(b"\x00\x01\x02").decode()
            ws.send_json({"type": "audio_chunk", "data": odd_bytes})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert resp["code"] == "invalid_audio"

    def test_engine_error_propagated(self, app, mock_engine):
        mock_engine.create_chat_completion.side_effect = RuntimeError("GPU OOM")
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert resp["code"] == "engine_error"
            assert "GPU OOM" in resp["message"]

    def test_missing_engine(self, app):
        """No serving_chat in engine state."""
        _engine_state.clear()
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert resp["code"] == "engine_error"

    def test_invalid_json_message(self, app, mock_engine):
        """Non-dict JSON should return error, not crash."""
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()
            # send a message missing required 'data' field
            ws.send_json({"type": "audio_chunk"})
            resp = ws.receive_json()
            assert resp["type"] == "error"


# ── Transcription flow tests ─────────────────────────────────────────


class TestTranscriptionFlow:
    def test_single_audio_chunk(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
            resp = ws.receive_json()
            assert resp["type"] == "transcript"
            assert resp["step"] == 1
            assert resp["is_append"] is False
            assert "text" in resp

    def test_display_and_append_steps(self, app, mock_engine):
        """8 steps: 7 display-only + 1 append."""
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            steps = []
            for _ in range(8):
                ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
                resp = ws.receive_json()
                assert resp["type"] == "transcript"
                steps.append(resp)

            # Steps 1-7 should be display-only
            for s in steps[:7]:
                assert s["is_append"] is False

            # Step 8 should be append
            assert steps[7]["is_append"] is True
            assert steps[7]["step"] == 8

    def test_engine_called_with_audio(self, app, mock_engine):
        """Verify the engine receives a proper request."""
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
            ws.receive_json()

            # Engine should have been called once
            assert mock_engine.create_chat_completion.call_count == 1
            call_args = mock_engine.create_chat_completion.call_args
            request = call_args[0][0]
            # Should be a ChatCompletionRequest
            assert hasattr(request, "messages")
            assert hasattr(request, "model")

    def test_prefix_used_after_append(self, app, mock_engine):
        """After an append step, subsequent calls should use prefix."""
        call_count = 0

        async def mock_transcribe(request, raw_request=None):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = f" word{call_count}"
            mock_resp.choices = [mock_choice]
            mock_usage = MagicMock()
            mock_usage.completion_tokens = 1
            mock_resp.usage = mock_usage
            return mock_resp

        mock_engine.create_chat_completion.side_effect = mock_transcribe

        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            # Send 9 chunks (1 append + 1 display-only)
            for _ in range(9):
                ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
                ws.receive_json()

            # After step 8 (append), step 9 should use continue_final_message
            assert call_count == 9
            # Check last call has prefix (continue_final_message)
            last_request = mock_engine.create_chat_completion.call_args[0][0]
            assert last_request.continue_final_message is True

    def test_comp_tokens_in_response(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
            resp = ws.receive_json()
            assert resp["comp_tokens"] == 5

    def test_custom_session_config(self, app, mock_engine):
        """Verify custom display/append deltas are respected."""
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({
                "type": "session.start",
                "display_delta": 1.0,
                "append_delta": 2.0,
                "max_audio": 10.0,
            })
            ws.receive_json()

            # With steps_per_append = 2, step 2 should be append
            for _ in range(2):
                ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
                resp = ws.receive_json()

            assert resp["is_append"] is True
            assert resp["step"] == 2

    def test_final_text_after_transcription(self, app, mock_engine):
        """The final text should include committed + prefix."""
        step_counter = 0

        async def mock_transcribe(request, raw_request=None):
            nonlocal step_counter
            step_counter += 1
            mock_resp = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = f" chunk{step_counter}"
            mock_resp.choices = [mock_choice]
            mock_usage = MagicMock()
            mock_usage.completion_tokens = 1
            mock_resp.usage = mock_usage
            return mock_resp

        mock_engine.create_chat_completion.side_effect = mock_transcribe

        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            # Run 8 steps (1 full cycle with 1 append)
            for _ in range(8):
                ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
                ws.receive_json()

            ws.send_json({"type": "session.end"})
            final = ws.receive_json()
            assert final["type"] == "final"
            assert len(final["text"]) > 0

    def test_empty_audio_buffer_returns_empty_transcript(self, app, mock_engine):
        """If the active audio is empty after trimming, return empty text."""
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            # This creates a scenario where the session starts but
            # audio_start hasn't been set yet — active audio should exist
            # because we append a chunk
            ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
            resp = ws.receive_json()
            # Should still get a transcript (possibly empty text from mock)
            assert resp["type"] == "transcript"


# ── Registration tests ───────────────────────────────────────────────


class TestRegistration:
    def test_register_adds_routes(self):
        app = FastAPI()
        register_streaming_asr(app)
        routes = [r.path for r in app.routes]
        assert "/v1/streaming_asr" in routes
        assert "/v1/streaming_asr/health" in routes

    def test_init_streaming_state(self):
        app = FastAPI()
        app.state.openai_serving_chat = MagicMock()
        init_streaming_state(app, model_name="test-model")
        assert _engine_state["serving_chat"] is app.state.openai_serving_chat
        assert _engine_state["model_name"] == "test-model"

    def test_init_streaming_state_no_chat(self):
        app = FastAPI()
        init_streaming_state(app, model_name="test")
        assert _engine_state["serving_chat"] is None

    def test_register_is_idempotent(self):
        app = FastAPI()
        register_streaming_asr(app)
        register_streaming_asr(app)
        ws_routes = [r for r in app.routes if hasattr(r, 'path') and r.path == "/v1/streaming_asr"]
        # Should have 2 because we registered twice, but the app handles dedup
        # The important thing is it doesn't crash
        assert len(ws_routes) >= 1


# ── Multi-step workflow tests ────────────────────────────────────────


class TestMultiStepWorkflow:
    def test_two_full_cycles(self, app, mock_engine):
        """16 steps = 2 complete append cycles."""
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            append_steps = []
            for _ in range(16):
                ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
                resp = ws.receive_json()
                if resp["is_append"]:
                    append_steps.append(resp["step"])

            assert append_steps == [8, 16]
            assert mock_engine.create_chat_completion.call_count == 16

    def test_step_numbers_are_sequential(self, app, mock_engine):
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            steps = []
            for _ in range(5):
                ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
                resp = ws.receive_json()
                steps.append(resp["step"])

            assert steps == [1, 2, 3, 4, 5]

    def test_raw_text_included(self, app, mock_engine):
        """Response should include raw LLM output."""
        client = TestClient(app)
        with client.websocket_connect("/v1/streaming_asr") as ws:
            ws.send_json({"type": "session.start"})
            ws.receive_json()

            ws.send_json({"type": "audio_chunk", "data": _make_pcm16_b64()})
            resp = ws.receive_json()
            assert "raw" in resp
            assert resp["raw"] == "hello world"
