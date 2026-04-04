"""Streaming ASR endpoint for MERaLiON2 vLLM plugin.

Provides a WebSocket-based streaming transcription endpoint that runs
inside the vLLM process, eliminating HTTP overhead and enabling
server-side audio buffering with mel-spectrogram caching.
"""
