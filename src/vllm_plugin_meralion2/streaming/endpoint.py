"""WebSocket endpoint for MERaLiON streaming ASR.

Runs inside the vLLM process and shares the engine with
``/v1/chat/completions``.  Avoids HTTP round-trip, base64 encoding,
and WAV parsing overhead.

Registration
~~~~~~~~~~~~
The endpoint is registered by monkey-patching ``build_app`` in
``vllm.entrypoints.openai.api_server`` from the plugin's
``register()`` function.  This adds ``/v1/streaming_asr`` alongside
the existing OpenAI-compatible routes.

Architecture
~~~~~~~~~~~~
::

    Client ──WebSocket──▶ /v1/streaming_asr
      sends 0.5 s PCM16 chunks
                          │
                   StreamingASRSession
                   (audio buffer, prefix)
                          │
                   Build ChatCompletionRequest
                   (audio as data-URL, prefix)
                          │
                   engine_client.generate()
                          │
                   ◀──────┘
      receives transcript JSON
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import struct
import time
import wave
from functools import partial
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

from .protocol import (
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
from .session import SessionConfig, StreamingASRSession, strip_speaker_tags

if TYPE_CHECKING:
    pass

logger = logging.getLogger("vllm_plugin.streaming_asr")

router = APIRouter()

# ── Global state (set by init_streaming_state) ───────────────────────

_engine_state: dict[str, Any] = {}
_sessions: dict[str, StreamingASRSession] = {}

SESSION_TIMEOUT_SEC = 120


# ── Audio encoding (numpy → base64 data URL) ────────────────────────

def _encode_audio_url(audio: np.ndarray, sr: int = 16_000) -> str:
    """Encode float32 audio as ``data:audio/wav;base64,...`` URL.

    This is needed because the vLLM chat completions handler expects
    audio as a data URL.  The encoding runs in-process so it is much
    faster than the external client path (no network transfer).
    """
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


# ── Transcription via vLLM engine (direct, no chat handler) ──────────

def _get_engine_client():
    """Lazily resolve the vLLM engine client."""
    client = _engine_state.get("engine_client")
    if client is not None:
        return client
    app = _engine_state.get("app")
    if app is not None:
        client = getattr(app.state, "engine_client", None)
        if client is not None:
            _engine_state["engine_client"] = client
            return client
    raise RuntimeError("Streaming ASR endpoint not initialized (no engine_client)")


def _get_tokenizer():
    """Lazily resolve and cache the tokenizer."""
    tok = _engine_state.get("_tokenizer")
    if tok is not None:
        return tok
    app = _engine_state.get("app")
    if app is None:
        raise RuntimeError("No app in engine state")
    serving = getattr(app.state, "openai_serving_chat", None)
    if serving is None:
        raise RuntimeError("No openai_serving_chat in app state")
    tok = serving.renderer.tokenizer
    _engine_state["_tokenizer"] = tok
    return tok


def _build_prompt_text(prefix: Optional[str]) -> str:
    """Build the full chat-templated prompt text for transcription.

    Uses the tokenizer's chat template to produce the exact format
    the model expects.  The ``<SpeechHere>`` placeholder is left in
    the text so vLLM's multimodal processor can find and expand it.

    Results are cached for the static portions.
    """
    tokenizer = _get_tokenizer()

    cache = _engine_state.get("_prompt_cache")
    if cache is None:
        prompt_template = _engine_state.get(
            "prompt_template", "Transcribe the following audio."
        )
        user_text = (
            f"Instruction: {prompt_template} \n"
            "Follow the text instruction based on the following audio: <SpeechHere>"
        )
        # Render with generation prompt (no prefix)
        messages_no_prefix = [{"role": "user", "content": user_text}]
        base_text = tokenizer.apply_chat_template(
            messages_no_prefix, tokenize=False, add_generation_prompt=True,
        )
        cache = {"base_text": base_text, "user_text": user_text}
        _engine_state["_prompt_cache"] = cache

    if prefix is None:
        return cache["base_text"]

    # With prefix: apply chat template with continue_final_message
    messages = [
        {"role": "user", "content": cache["user_text"]},
        {"role": "assistant", "content": prefix},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )


async def _transcribe_via_engine(
    audio: np.ndarray,
    prefix: Optional[str],
    *,
    sr: int = 16_000,
    model: str = "",
    prompt_template: str = "Transcribe the following audio.",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> tuple[str, int]:
    """Call the vLLM engine directly with a TokensPrompt.

    Bypasses the chat completion handler entirely — no JSON parsing,
    no base64 encoding, no WAV wrapping. Sends raw numpy audio +
    pre-built token IDs directly to ``engine_client.generate()``.

    Returns ``(text, completion_tokens)``.
    """
    from vllm.inputs import TextPrompt
    from vllm.sampling_params import SamplingParams

    engine = _get_engine_client()

    prompt_text = _build_prompt_text(prefix)

    prompt = TextPrompt(
        prompt=prompt_text,
        multi_modal_data={"audio": audio},
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    request_id = f"stream-asr-{uuid4()}"

    full_text = ""
    comp_tokens = 0
    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        if output.outputs:
            full_text = output.outputs[0].text
            comp_tokens = len(output.outputs[0].token_ids)

    return full_text, comp_tokens


# ── WebSocket handler ────────────────────────────────────────────────

@router.websocket("/v1/streaming_asr")
async def streaming_asr_endpoint(ws: WebSocket) -> None:
    """WebSocket endpoint for streaming ASR transcription."""
    await ws.accept()
    session: Optional[StreamingASRSession] = None
    sid = ""

    try:
        while True:
            raw_msg = await ws.receive_json()
            try:
                msg = parse_client_message(raw_msg)
            except ValueError as exc:
                await ws.send_json(
                    ErrorResponse(message=str(exc), code="invalid_message").model_dump()
                )
                continue

            # ── session.start ────────────────────────────────────
            if isinstance(msg, SessionStartMessage):
                config = SessionConfig(
                    display_delta=msg.display_delta,
                    append_delta=msg.append_delta,
                    max_audio=msg.max_audio,
                )
                session = StreamingASRSession(config=config)
                sid = session.session_id
                _sessions[sid] = session
                logger.info("Session %s started", sid)
                await ws.send_json(
                    SessionStartedResponse(session_id=sid).model_dump()
                )
                continue

            # ── session.end ──────────────────────────────────────
            if isinstance(msg, SessionEndMessage):
                if session is None:
                    await ws.send_json(
                        ErrorResponse(
                            message="No active session", code="no_session"
                        ).model_dump()
                    )
                    continue
                final_text = session.get_final_text()
                await ws.send_json(FinalResponse(text=final_text).model_dump())
                logger.info(
                    "Session %s ended (%d samples)",
                    sid,
                    session.total_samples,
                )
                _sessions.pop(sid, None)
                break

            # ── audio_chunk ──────────────────────────────────────
            # Append audio to server-side buffer, then optionally
            # transcribe.  The client controls all streaming logic
            # (prefix, step types, display cleaning).
            if isinstance(msg, AudioChunkMessage):
                if session is None:
                    await ws.send_json(
                        ErrorResponse(
                            message="Send session.start first",
                            code="no_session",
                        ).model_dump()
                    )
                    continue

                try:
                    audio_f32 = decode_pcm16(msg.data)
                except ValueError as exc:
                    await ws.send_json(
                        ErrorResponse(
                            message=f"Bad audio: {exc}", code="invalid_audio"
                        ).model_dump()
                    )
                    continue

                session.append_audio(audio_f32)

                # If the message includes a "transcribe" flag, run
                # transcription on the current buffer.  Otherwise
                # just acknowledge the append (for batching chunks).
                prefix = raw_msg.get("prefix")  # client-managed prefix
                audio_start = raw_msg.get("audio_start", 0.0)  # client-managed trim point
                start_sample = int(audio_start * session.config.sample_rate)

                active_audio = session.audio_buffer[start_sample:] \
                    if start_sample < session.total_samples \
                    else np.array([], dtype=np.float32)

                if len(active_audio) == 0:
                    await ws.send_json({
                        "type": "transcript",
                        "text": "",
                        "comp_tokens": 0,
                    })
                    continue

                try:
                    raw_text, comp_tokens = await _transcribe_via_engine(
                        active_audio, prefix,
                    )
                except Exception as exc:
                    logger.error("Session %s engine error: %s", sid, exc)
                    await ws.send_json(
                        ErrorResponse(
                            message=f"Engine error: {exc}",
                            code="engine_error",
                        ).model_dump()
                    )
                    continue

                await ws.send_json({
                    "type": "transcript",
                    "text": raw_text,
                    "comp_tokens": comp_tokens,
                })

    except WebSocketDisconnect:
        logger.info("Session %s disconnected", sid)
    except Exception:
        logger.exception("Session %s unexpected error", sid)
    finally:
        if sid:
            _sessions.pop(sid, None)


# ── Health endpoint ──────────────────────────────────────────────────

# ── REST endpoint: direct engine transcription (no chat handler) ─────

@router.post("/v1/transcribe_direct")
async def transcribe_direct_endpoint(request_body: dict) -> dict:
    """Direct-to-engine transcription endpoint.

    Accepts JSON with ``audio_url`` (data:audio/wav;base64,...) and
    optional ``prefix`` for continuation.  Calls the engine directly
    via ``TextPrompt``, bypassing the chat completion handler.

    This avoids the event-loop serialization of JSON/base64/WAV parsing
    that causes ~150ms gap at 8 concurrent workers.
    """
    audio_url = request_body.get("audio_url", "")
    prefix = request_body.get("prefix")
    temperature = request_body.get("temperature", 0.0)
    max_tokens = request_body.get("max_tokens", 512)

    if not audio_url:
        return {"error": "Missing audio_url"}

    # Decode the data URL to numpy array (we do this once, in-process)
    try:
        audio = _decode_audio_url(audio_url)
    except Exception as exc:
        return {"error": f"Bad audio: {exc}"}

    try:
        text, comp_tokens = await _transcribe_via_engine(
            audio, prefix, temperature=temperature, max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.error("transcribe_direct error: %s", exc)
        return {"error": str(exc)}

    return {"text": text, "completion_tokens": comp_tokens}


def _decode_audio_url(audio_url: str) -> np.ndarray:
    """Decode a data:audio/wav;base64,... URL to float32 numpy array."""
    import wave

    b64_data = audio_url.split(",", 1)[1]
    wav_bytes = base64.b64decode(b64_data)
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return pcm


@router.get("/v1/streaming_asr/health")
async def streaming_asr_health() -> dict:
    """Health check for the streaming ASR endpoint."""
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
        "engine_ready": "serving_chat" in _engine_state,
    }


# ── Session reaper ───────────────────────────────────────────────────

async def _reap_idle_sessions() -> None:
    """Background task to clean up idle sessions."""
    while True:
        await asyncio.sleep(15)
        now = time.time()
        expired = [
            sid
            for sid, s in _sessions.items()
            if now - s.last_active > SESSION_TIMEOUT_SEC
        ]
        for sid in expired:
            _sessions.pop(sid, None)
            logger.info("Reaped idle session %s", sid)


# ── App registration ─────────────────────────────────────────────────

_reaper_task: Optional[asyncio.Task] = None


def register_streaming_asr(app: FastAPI) -> None:
    """Attach the streaming ASR router to the vLLM FastAPI app."""
    app.include_router(router)
    logger.info("Streaming ASR endpoint registered at /v1/streaming_asr")


def init_streaming_state(
    app: FastAPI,
    *,
    model_name: str = "",
    prompt_template: str = "Transcribe the following audio.",
) -> None:
    """Initialize streaming ASR state after the engine is ready.

    Called from the patched ``build_app`` after the engine client and
    OpenAI serving handlers have been created.
    """
    global _reaper_task

    serving_chat = getattr(app.state, "openai_serving_chat", None)
    _engine_state["serving_chat"] = serving_chat
    _engine_state["app"] = app
    _engine_state["model_name"] = model_name
    _engine_state["prompt_template"] = prompt_template

    if _reaper_task is None or _reaper_task.done():
        _reaper_task = asyncio.ensure_future(_reap_idle_sessions())

    logger.info(
        "Streaming ASR state initialized (model=%s, engine=%s)",
        model_name,
        "ready" if serving_chat is not None else "MISSING",
    )
