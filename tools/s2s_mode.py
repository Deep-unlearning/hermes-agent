"""Realtime speech-to-speech transport for the Hermes CLI.

This module connects the interactive CLI to a speech-to-speech server that
implements the OpenAI Realtime protocol.  The voice model acts as a small,
spoken control plane: it can answer lightweight conversation itself, delegate
computer work to the current Hermes session, inspect live progress, steer or
stop the active turn, and start an independent background task.  Computer work
still goes through Hermes so normal tool output, approvals, and session history
remain visible in the terminal.

The heavy audio dependency is optional and imported only when the mode starts.
Install it with ``pip install hermes-agent[voice]``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


def _safe_spoken_text(value: Any) -> str:
    """Force-redact secrets at the audio egress boundary."""
    text = str(value or "")
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(
            text, force=True, redact_url_credentials=True
        )
    except Exception:
        logger.debug("Could not redact S2S speech output", exc_info=True)
        return "Hermes produced output that could not be safely prepared for speech."


S2S_INSTRUCTIONS = """\
You are the realtime spoken controller for Hermes, the computer agent running
in this terminal. Decide how to handle each utterance:

- Answer lightweight conversation yourself when no terminal, repository, web,
  private session state, or computer action is needed.
- For coding, files, shell commands, tests, research, or any other computer
  work, call send_to_hermes once with a faithful and complete instruction.
- For a progress question such as "what are you doing?" or "give me an update",
  call get_hermes_status. Never invent Hermes progress.
- When the user wants to modify the active task without cancelling it, call
  steer_hermes. This corresponds to Hermes' /steer command.
- When the user explicitly says "in the background", "by the way", or asks for
  /btw, call start_background_task. This corresponds to /background or /btw and
  runs in an independent session.
- When the user explicitly wants the active Hermes task cancelled, call
  stop_hermes. Merely interrupting your speech must not cancel Hermes.
- If Hermes is waiting for an approval or clarification, relay the user's exact
  answer through send_to_hermes. Never approve an action on the user's behalf.

Useful command context: status/update checks progress; steer changes the active
task after its next tool call; background/btw starts parallel work; stop cancels
the foreground task. A new send_to_hermes request is queued when Hermes is busy.
Never claim that a command ran until Hermes reports its result. Never request or
repeat passwords, API keys, or other secrets; secure entry stays in the terminal.
After a tool result arrives, speak it faithfully and conversationally. Keep
spoken replies concise while preserving important facts, warnings, and errors.
"""

S2S_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "send_to_hermes",
        "description": (
            "Delegate coding, repository, file, shell, test, research, or other "
            "computer work to Hermes in the current terminal session. Hermes "
            "retains its normal command approvals. The tool acknowledges "
            "immediately and announces the result when Hermes finishes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "A faithful, complete version of the user's request.",
                }
            },
            "required": ["message"],
        },
    },
    {
        "type": "function",
        "name": "get_hermes_status",
        "description": (
            "Get a truthful live progress update for the foreground Hermes turn, "
            "including its current activity, elapsed time, prompts, and background "
            "task count. Use this for status or progress questions."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "steer_hermes",
        "description": (
            "Add guidance to the currently running Hermes turn without cancelling "
            "it. The guidance is injected after Hermes' next tool call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The new guidance for the active Hermes task.",
                }
            },
            "required": ["message"],
        },
    },
    {
        "type": "function",
        "name": "stop_hermes",
        "description": (
            "Cancel the currently running foreground Hermes task. Use only when "
            "the user explicitly wants the task stopped, not merely the speech."
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "type": "function",
        "name": "start_background_task",
        "description": (
            "Run a request in an independent Hermes background session, like "
            "/background or /btw. Use only when parallel/background work is wanted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The complete task to run in the background.",
                }
            },
            "required": ["message"],
        },
    },
]


@dataclass(frozen=True)
class S2STurn:
    """A voice-originated CLI turn and its Realtime function call identity."""

    message: str
    call_id: str


@dataclass(frozen=True)
class S2STurnAck:
    """Immediate tool result plus whether a later Hermes result will arrive."""

    message: str
    await_result: bool = True


@dataclass(frozen=True)
class S2SControlCall:
    """A voice-side control action that does not become a normal CLI turn."""

    name: str
    call_id: str
    message: str = ""


@dataclass(frozen=True)
class _S2SOutbound:
    kind: str  # "tool_ack" | "hermes_result"
    call_id: str
    text: str


@dataclass(frozen=True)
class S2SConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    model: str = "local"
    voice: str | None = None
    send_rate: int = 16000
    recv_rate: int = 16000
    chunk_size: int = 1024
    input_device: int | None = None
    output_device: int | None = None
    block_mic_during_playback: bool = False
    max_spoken_chars: int = 4000
    connect_timeout: float = 5.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "S2SConfig":
        cfg = raw if isinstance(raw, Mapping) else {}

        def integer(
            name: str,
            default: int,
            *,
            minimum: int = 0,
            maximum: int | None = None,
        ) -> int:
            value = cfg.get(name, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
                or (maximum is not None and value > maximum)
            ):
                return default
            return value

        def device(name: str) -> int | None:
            value = cfg.get(name)
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        host = str(cfg.get("host") or cls.host).strip()
        model = str(cfg.get("model") or cls.model).strip()
        voice_raw = cfg.get("voice")
        voice = str(voice_raw).strip() if voice_raw else None
        timeout_raw = cfg.get("connect_timeout", cls.connect_timeout)
        timeout = (
            float(timeout_raw)
            if isinstance(timeout_raw, (int, float))
            and not isinstance(timeout_raw, bool)
            and timeout_raw > 0
            else cls.connect_timeout
        )
        block_mic_raw = cfg.get("block_mic_during_playback", False)
        return cls(
            host=host or cls.host,
            port=integer("port", cls.port, minimum=1, maximum=65535),
            model=model or cls.model,
            voice=voice,
            send_rate=integer("send_rate", cls.send_rate, minimum=1),
            recv_rate=integer("recv_rate", cls.recv_rate, minimum=1),
            chunk_size=integer("chunk_size", cls.chunk_size, minimum=1),
            input_device=device("input_device"),
            output_device=device("output_device"),
            block_mic_during_playback=(
                block_mic_raw if isinstance(block_mic_raw, bool) else False
            ),
            max_spoken_chars=integer("max_spoken_chars", cls.max_spoken_chars, minimum=1),
            connect_timeout=timeout,
        )


def check_s2s_requirements(config: S2SConfig, *, probe_server: bool = True) -> dict[str, Any]:
    """Return a user-facing availability report without retaining audio imports."""
    missing: list[str] = []
    details: list[str] = []
    try:
        import sounddevice  # noqa: F401
    except (ImportError, OSError) as exc:
        missing.append("sounddevice")
        details.append(f"Audio input/output unavailable: {exc}")

    if config.send_rate not in {16000, 24000}:
        details.append("send_rate must be 16000 (local default) or 24000")
    if config.recv_rate not in {16000, 24000}:
        details.append("recv_rate must be 16000 (local default) or 24000")

    if probe_server and not missing and not details:
        try:
            with socket.create_connection(
                (config.host, config.port), timeout=min(config.connect_timeout, 2.0)
            ):
                pass
        except OSError as exc:
            details.append(
                f"Cannot reach the S2S server at {config.host}:{config.port}: {exc}"
            )

    return {
        "available": not missing and not details,
        "missing_packages": missing,
        "details": "\n".join(details) if details else "S2S client requirements are available.",
    }


def _pcm_format(rate: int) -> dict[str, Any] | None:
    if rate == 16000:
        return None
    if rate == 24000:
        return {"type": "audio/pcm", "rate": 24000}
    raise ValueError("S2S sample rates must be 16000 or 24000 Hz")


def build_session_update(config: S2SConfig) -> dict[str, Any]:
    input_cfg: dict[str, Any] = {
        "turn_detection": {"type": "server_vad", "interrupt_response": True}
    }
    output_cfg: dict[str, Any] = {}
    input_format = _pcm_format(config.send_rate)
    output_format = _pcm_format(config.recv_rate)
    if input_format is not None:
        input_cfg["format"] = input_format
    if output_format is not None:
        output_cfg["format"] = output_format
    if config.voice:
        output_cfg["voice"] = config.voice

    audio: dict[str, Any] = {"input": input_cfg}
    if output_cfg:
        audio["output"] = output_cfg
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": S2S_INSTRUCTIONS,
            "tools": S2S_TOOLS,
            "tool_choice": "auto",
            "audio": audio,
        },
    }


def decode_hermes_turn(name: str, arguments: str | None, call_id: str) -> S2STurn | None:
    """Decode the delegation tool exposed to the voice model."""
    if name != "send_to_hermes" or not call_id:
        return None
    try:
        data = json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    message = str(data.get("message") or "").strip() if isinstance(data, dict) else ""
    return S2STurn(message=message, call_id=call_id) if message else None


_S2S_CONTROL_TOOLS = {
    "get_hermes_status",
    "steer_hermes",
    "stop_hermes",
    "start_background_task",
}


def decode_control_call(
    name: str, arguments: str | None, call_id: str
) -> S2SControlCall | None:
    """Decode one of the scoped controls exposed only to the voice model."""
    if name not in _S2S_CONTROL_TOOLS or not call_id:
        return None
    try:
        data = json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    message = str(data.get("message") or "").strip()
    if name in {"steer_hermes", "start_background_task"} and not message:
        return None
    return S2SControlCall(name=name, call_id=call_id, message=message)


TurnCallback = Callable[[S2STurn], S2STurnAck | None]
ControlCallback = Callable[[S2SControlCall], S2STurnAck | None]
TranscriptCallback = Callable[[str, bool], None]
StateCallback = Callable[[str, str], None]


class S2SMode:
    """Thread-owned Realtime connection and full-duplex audio streams."""

    def __init__(
        self,
        config: S2SConfig,
        *,
        on_turn: TurnCallback,
        on_control: ControlCallback | None = None,
        on_transcript: TranscriptCallback | None = None,
        on_state: StateCallback | None = None,
    ) -> None:
        self.config = config
        self._on_turn = on_turn
        self._on_control = on_control
        self._on_transcript = on_transcript
        self._on_state = on_state
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._responses: Queue[_S2SOutbound] = Queue()
        self._mic_queue: Queue[bytes] = Queue(maxsize=128)
        self._playback = bytearray()
        self._playback_lock = threading.Lock()
        self._speaker_active_until = 0.0
        self._seen_call_ids: set[str] = set()
        self._pending_results: set[str] = set()
        self._acknowledged_call_ids: set[str] = set()
        self._early_results: dict[str, str] = {}
        self._result_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return bool(
            self._thread
            and self._thread.is_alive()
            and self._ready.is_set()
            and not self._stop.is_set()
            and self._error is None
        )

    @property
    def error(self) -> str:
        return str(self._error) if self._error else ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._error = None
        self._emit_state("connecting", f"{self.config.host}:{self.config.port}")
        self._thread = threading.Thread(target=self._thread_main, daemon=True, name="hermes-s2s")
        self._thread.start()
        if not self._ready.wait(timeout=self.config.connect_timeout):
            self.stop()
            raise RuntimeError(
                f"Timed out connecting to the S2S server at "
                f"{self.config.host}:{self.config.port}"
            )
        if self._error is not None:
            error = self._error
            self.stop()
            raise RuntimeError(f"Could not start S2S mode: {error}") from error

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        self._clear_playback()

    def deliver_response(self, call_id: str, response: str | None) -> None:
        """Queue a completed Hermes result as a new spoken announcement."""
        if not call_id:
            return
        text = _safe_spoken_text(
            response or "Hermes did not return a response."
        ).strip()
        if len(text) > self.config.max_spoken_chars:
            text = text[: self.config.max_spoken_chars].rstrip() + " …(truncated)"
        with self._result_lock:
            if call_id not in self._acknowledged_call_ids:
                self._early_results[call_id] = text
                return
        self._responses.put(_S2SOutbound(kind="hermes_result", call_id=call_id, text=text))

    def _emit_state(self, state: str, detail: str = "") -> None:
        if self._on_state:
            try:
                self._on_state(state, detail)
            except Exception:
                logger.debug("S2S state callback failed", exc_info=True)

    def _emit_transcript(self, text: str, final: bool) -> None:
        if self._on_transcript:
            try:
                self._on_transcript(text, final)
            except Exception:
                logger.debug("S2S transcript callback failed", exc_info=True)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as exc:
            self._error = exc
            self._ready.set()
            self._emit_state("error", str(exc))
            logger.warning("S2S mode stopped with an error: %s", exc)
        finally:
            self._stop.set()
            if self._error is None:
                self._emit_state("stopped", "")

    def _clear_playback(self) -> None:
        self._speaker_active_until = 0.0
        with self._playback_lock:
            self._playback.clear()

    def _open_streams(self):
        import sounddevice as sd

        def on_mic(indata, _frames, _time_info, status):
            if status:
                logger.debug("S2S microphone status: %s", status)
            if self.config.block_mic_during_playback:
                with self._playback_lock:
                    playing = bool(self._playback)
                if playing or time.monotonic() < self._speaker_active_until:
                    return
            try:
                self._mic_queue.put_nowait(bytes(indata))
            except Exception:
                pass

        def on_speaker(outdata, _frames, _time_info, status):
            if status:
                logger.debug("S2S speaker status: %s", status)
            needed = len(outdata)
            with self._playback_lock:
                available = min(needed, len(self._playback))
                if available:
                    outdata[:available] = self._playback[:available]
                    del self._playback[:available]
                if available < needed:
                    outdata[available:] = b"\x00" * (needed - available)

        mic = sd.RawInputStream(
            samplerate=self.config.send_rate,
            channels=1,
            dtype="int16",
            blocksize=self.config.chunk_size,
            callback=on_mic,
            device=self.config.input_device,
        )
        speaker = sd.RawOutputStream(
            samplerate=self.config.recv_rate,
            channels=1,
            dtype="int16",
            blocksize=self.config.chunk_size,
            callback=on_speaker,
            device=self.config.output_device,
        )
        return mic, speaker

    async def _run(self) -> None:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key="local-s2s",
            base_url=f"http://{self.config.host}:{self.config.port}/v1",
            websocket_base_url=f"ws://{self.config.host}:{self.config.port}/v1",
        )
        mic = speaker = None
        try:
            async with client.realtime.connect(model=self.config.model) as conn:
                await conn.send(build_session_update(self.config))
                mic, speaker = self._open_streams()
                mic.start()
                speaker.start()
                self._ready.set()
                self._emit_state("listening", "")

                response_idle = asyncio.Event()
                response_idle.set()
                tasks = {
                    asyncio.create_task(self._send_audio(conn)),
                    asyncio.create_task(self._receive_events(conn, response_idle)),
                    asyncio.create_task(self._send_responses(conn, response_idle)),
                    asyncio.create_task(self._wait_for_stop()),
                }
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                self._stop.set()
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    if not task.cancelled() and task.exception() is not None:
                        raise task.exception()
        finally:
            self._stop.set()
            for stream in (mic, speaker):
                if stream is None:
                    continue
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            await client.close()

    async def _wait_for_stop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.1)

    async def _send_audio(self, conn) -> None:
        while not self._stop.is_set():
            try:
                chunk = await asyncio.to_thread(self._mic_queue.get, True, 0.1)
            except Empty:
                continue
            await conn.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
            )

    async def _send_responses(self, conn, response_idle: asyncio.Event) -> None:
        while not self._stop.is_set():
            try:
                outbound = await asyncio.to_thread(self._responses.get, True, 0.1)
            except Empty:
                continue
            await response_idle.wait()
            if outbound.kind == "tool_ack":
                item = {
                    "type": "function_call_output",
                    "call_id": outbound.call_id,
                    "output": outbound.text,
                }
                instructions = (
                    "Briefly acknowledge the tool result to the user. "
                    "Do not call another tool."
                )
            else:
                item = {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "[HERMES RESULT — automated completion, not a new user request] "
                                + outbound.text
                            ),
                        }
                    ],
                }
                instructions = (
                    "Speak the automated Hermes result faithfully and concisely. "
                    "Do not call another tool."
                )
            await conn.send(
                {"type": "conversation.item.create", "item": item}
            )
            response_idle.clear()
            await conn.send(
                {
                    "type": "response.create",
                    "response": {
                        "tool_choice": "none",
                        "instructions": instructions,
                    },
                }
            )
            if outbound.kind == "hermes_result":
                self._pending_results.discard(outbound.call_id)

    async def _receive_events(self, conn, response_idle: asyncio.Event) -> None:
        while not self._stop.is_set():
            event = await conn.recv()
            event_type = event.type
            if event_type == "input_audio_buffer.speech_started":
                self._clear_playback()
                self._emit_transcript("", False)
                self._emit_state("listening", "")
            elif event_type == "conversation.item.input_audio_transcription.delta":
                self._emit_transcript(event.delta.strip(), False)
            elif event_type == "conversation.item.input_audio_transcription.completed":
                self._emit_transcript(event.transcript.strip(), True)
            elif event_type == "response.created":
                response_idle.clear()
            elif event_type == "response.output_audio.delta":
                audio = base64.b64decode(event.delta)
                with self._playback_lock:
                    self._playback.extend(audio)
                self._speaker_active_until = time.monotonic() + max(
                    0.15, len(audio) / (2 * self.config.recv_rate)
                )
                self._emit_state("speaking", "")
            elif event_type == "response.function_call_arguments.done":
                turn = decode_hermes_turn(event.name, event.arguments, event.call_id)
                control = decode_control_call(
                    event.name, event.arguments, event.call_id
                )
                call = turn or control
                if call and call.call_id not in self._seen_call_ids:
                    self._seen_call_ids.add(call.call_id)
                    try:
                        if turn is not None:
                            ack = self._on_turn(turn)
                        elif self._on_control is not None:
                            ack = self._on_control(control)
                        else:
                            ack = S2STurnAck(
                                "That Hermes voice control is unavailable.",
                                await_result=False,
                            )
                    except Exception as exc:
                        logger.exception("Could not route S2S tool call to the CLI")
                        ack = S2STurnAck(
                            f"Hermes could not perform that action: {exc}",
                            await_result=False,
                        )
                    if ack is None:
                        ack = S2STurnAck(
                            "Hermes accepted the voice action.", await_result=False
                        )
                    if ack.await_result:
                        self._pending_results.add(call.call_id)
                    self._responses.put(
                        _S2SOutbound(
                            kind="tool_ack",
                            call_id=call.call_id,
                            text=_safe_spoken_text(ack.message),
                        )
                    )
                    with self._result_lock:
                        self._acknowledged_call_ids.add(call.call_id)
                        early_result = self._early_results.pop(call.call_id, None)
                    if early_result is not None:
                        self._responses.put(
                            _S2SOutbound(
                                kind="hermes_result",
                                call_id=call.call_id,
                                text=early_result,
                            )
                        )
            elif event_type == "response.done":
                if event.response.status == "cancelled":
                    self._clear_playback()
                response_idle.set()
                self._emit_state("working" if self._pending_results else "listening", "")
            elif event_type == "error":
                response_idle.set()
                message = f"{event.error.type}: {event.error.message}"
                self._emit_state("error", message)
                logger.warning("S2S server error: %s", message)
