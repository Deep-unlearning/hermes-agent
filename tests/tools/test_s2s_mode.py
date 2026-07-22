"""Behavior tests for the native CLI realtime S2S transport."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.s2s_mode import (
    S2SConfig,
    S2SControlCall,
    S2SMode,
    S2STurn,
    S2STurnAck,
    build_session_update,
    check_s2s_requirements,
    decode_control_call,
    decode_hermes_turn,
)


def test_config_is_shape_safe_and_bounds_invalid_port():
    config = S2SConfig.from_mapping(
        {
            "host": "  localhost ",
            "port": 70000,
            "voice": " af_heart ",
            "send_rate": True,
            "recv_rate": 24000,
            "input_device": False,
            "block_mic_during_playback": True,
            "progress_interval": 1,
            "reconnect_enabled": "yes",
            "reconnect_attempts": -1,
            "reconnect_initial_delay": 8,
            "reconnect_max_delay": 2,
        }
    )

    assert config.host == "localhost"
    assert config.port == 8765
    assert config.voice == "af_heart"
    assert config.send_rate == 16000
    assert config.recv_rate == 24000
    assert config.input_device is None
    assert config.block_mic_during_playback is True
    assert config.progress_interval == 5.0
    assert config.reconnect_enabled is True
    assert config.reconnect_attempts == 0
    assert config.reconnect_initial_delay == 8.0
    assert config.reconnect_max_delay == 8.0


def test_session_update_exposes_scoped_voice_controller_tools():
    event = build_session_update(S2SConfig(voice="af_heart", recv_rate=24000))
    session = event["session"]

    assert event["type"] == "session.update"
    assert [tool["name"] for tool in session["tools"]] == [
        "send_to_hermes",
        "get_hermes_status",
        "steer_hermes",
        "queue_hermes_task",
        "stop_hermes",
        "start_background_task",
        "get_background_tasks",
        "stop_background_task",
        "steer_background_task",
    ]
    assert "shell commands" in session["instructions"]
    assert "Never invent Hermes progress" in session["instructions"]
    assert "/btw" in session["instructions"]
    assert "/queue" in session["instructions"]
    assert "Run a prompt in the background" in session["instructions"]
    assert session["audio"]["input"]["turn_detection"]["interrupt_response"] is True
    assert session["audio"]["output"] == {
        "format": {"type": "audio/pcm", "rate": 24000},
        "voice": "af_heart",
    }


def test_decode_hermes_turn_rejects_unknown_or_malformed_calls():
    assert decode_hermes_turn("other", "{}", "call-1") is None
    assert decode_hermes_turn("send_to_hermes", "not-json", "call-1") is None
    assert decode_hermes_turn("send_to_hermes", '{"message": "  "}', "call-1") is None

    assert decode_hermes_turn(
        "send_to_hermes", '{"message": " inspect the repo "}', "call-1"
    ) == S2STurn(message="inspect the repo", call_id="call-1")


def test_decode_control_call_validates_names_and_required_messages():
    assert decode_control_call("unknown", "{}", "call-1") is None
    assert decode_control_call("steer_hermes", "{}", "call-1") is None
    assert decode_control_call("get_hermes_status", "not-json", "call-1") is None

    assert decode_control_call(
        "get_hermes_status", "{}", "call-2"
    ) == S2SControlCall("get_hermes_status", "call-2")
    assert decode_control_call(
        "start_background_task", '{"message": " run tests "}', "call-3"
    ) == S2SControlCall("start_background_task", "call-3", "run tests")
    assert decode_control_call("stop_background_task", "{}", "call-4") is None
    assert decode_control_call("queue_hermes_task", "{}", "call-queue") is None
    assert decode_control_call(
        "stop_background_task", '{"task_id": " task one "}', "call-4"
    ) == S2SControlCall(
        "stop_background_task", "call-4", task_id="task one"
    )
    assert decode_control_call(
        "steer_background_task", '{"task_id": "1"}', "call-5"
    ) is None
    assert decode_control_call(
        "steer_background_task",
        '{"task_id": "1", "message": "focus on errors"}',
        "call-5",
    ) == S2SControlCall(
        "steer_background_task",
        "call-5",
        message="focus on errors",
        task_id="1",
    )


def test_deliver_response_truncates_at_configured_spoken_limit():
    mode = S2SMode(S2SConfig(max_spoken_chars=8), on_turn=MagicMock())
    mode._acknowledged_call_ids.add("call-1")
    mode.deliver_response("call-1", "abcdefghijk")

    outbound = mode._responses.get_nowait()
    assert outbound.kind == "hermes_result"
    assert outbound.call_id == "call-1"
    assert outbound.text == "abcdefgh …(truncated)"


def test_deliver_response_force_redacts_secrets_before_speech():
    mode = S2SMode(S2SConfig(), on_turn=MagicMock())
    mode._acknowledged_call_ids.add("call-secret")
    secret = "sk-proj-" + ("a" * 32) + "1234"

    mode.deliver_response("call-secret", f"Use {secret} for the request")

    outbound = mode._responses.get_nowait()
    assert secret not in outbound.text
    assert "sk-pro...1234" in outbound.text


def test_tool_call_gets_immediate_ack_and_tracks_later_result():
    accepted = []
    mode = S2SMode(
        S2SConfig(),
        on_turn=lambda turn: accepted.append(turn) or S2STurnAck("Working now."),
    )
    event = SimpleNamespace(
        type="response.function_call_arguments.done",
        name="send_to_hermes",
        arguments='{"message": "inspect the repo"}',
        call_id="call-2",
    )

    class FakeConn:
        async def recv(self):
            mode._stop.set()
            return event

    response_idle = asyncio.Event()
    response_idle.set()
    asyncio.run(mode._receive_events(FakeConn(), response_idle))

    assert accepted == [S2STurn("inspect the repo", "call-2")]
    outbound = mode._responses.get_nowait()
    assert outbound.kind == "tool_ack"
    assert outbound.call_id == "call-2"
    assert outbound.text == "Working now."
    assert mode._pending_results == {"call-2"}


def test_control_call_gets_immediate_result_without_becoming_a_cli_turn():
    accepted = []
    mode = S2SMode(
        S2SConfig(),
        on_turn=MagicMock(),
        on_control=lambda call: accepted.append(call)
        or S2STurnAck("Hermes is running tests.", await_result=False),
    )
    event = SimpleNamespace(
        type="response.function_call_arguments.done",
        name="get_hermes_status",
        arguments="{}",
        call_id="control-1",
    )

    class FakeConn:
        async def recv(self):
            mode._stop.set()
            return event

    response_idle = asyncio.Event()
    response_idle.set()
    asyncio.run(mode._receive_events(FakeConn(), response_idle))

    assert accepted == [S2SControlCall("get_hermes_status", "control-1")]
    outbound = mode._responses.get_nowait()
    assert outbound.kind == "tool_ack"
    assert outbound.text == "Hermes is running tests."
    assert mode._pending_results == set()


def test_result_that_finishes_during_callback_is_sent_after_tool_ack():
    mode = None

    def start_background(call):
        mode.deliver_response(call.call_id, "Background result")
        return S2STurnAck("Background task started.", await_result=True)

    mode = S2SMode(
        S2SConfig(), on_turn=MagicMock(), on_control=start_background
    )
    event = SimpleNamespace(
        type="response.function_call_arguments.done",
        name="start_background_task",
        arguments='{"message": "quick check"}',
        call_id="background-1",
    )

    class FakeConn:
        async def recv(self):
            mode._stop.set()
            return event

    response_idle = asyncio.Event()
    response_idle.set()
    asyncio.run(mode._receive_events(FakeConn(), response_idle))

    acknowledgement = mode._responses.get_nowait()
    completion = mode._responses.get_nowait()
    assert (acknowledgement.kind, acknowledgement.text) == (
        "tool_ack",
        "Background task started.",
    )
    assert (completion.kind, completion.text) == (
        "hermes_result",
        "Background result",
    )


def test_progress_callback_queues_one_announcement_after_interval():
    mode = S2SMode(
        S2SConfig(progress_interval=5),
        on_turn=MagicMock(),
        on_progress=lambda: "Hermes is running pytest.",
    )
    sleep_count = 0

    async def advance_clock(_delay):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 2:
            mode._stop.set()

    fake_time = SimpleNamespace(monotonic=MagicMock(side_effect=[100.0, 106.0]))
    with patch("tools.s2s_mode.asyncio.sleep", side_effect=advance_clock), patch(
        "tools.s2s_mode.time", fake_time
    ):
        asyncio.run(mode._announce_progress())

    outbound = mode._responses.get_nowait()
    assert outbound.kind == "progress"
    assert outbound.text == "Hermes is running pytest."


def test_progress_is_refreshed_before_it_is_spoken():
    sent = []
    mode = S2SMode(
        S2SConfig(),
        on_turn=MagicMock(),
        on_progress=lambda: "Fresh progress from the CLI.",
    )
    mode._responses.put(
        SimpleNamespace(kind="progress", call_id="", text="Stale progress")
    )
    mode._progress_pending = True

    class FakeConn:
        async def send(self, payload):
            sent.append(payload)
            if payload["type"] == "response.create":
                mode._stop.set()

    response_idle = asyncio.Event()
    response_idle.set()
    asyncio.run(mode._send_responses(FakeConn(), response_idle))

    assert "Fresh progress from the CLI" in sent[0]["item"]["content"][0]["text"]
    assert sent[1]["response"]["tool_choice"] == "none"
    assert mode._progress_pending is False


def test_connection_slot_failure_retries_and_recovers():
    states = []
    mode = S2SMode(
        S2SConfig(
            reconnect_enabled=True,
            reconnect_attempts=2,
            reconnect_initial_delay=0.001,
            reconnect_max_delay=0.001,
        ),
        on_turn=MagicMock(),
        on_state=lambda state, detail: states.append((state, detail)),
    )
    attempts = 0

    async def connect():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("All session slots are in use")
        mode._ready.set()
        mode._connected.set()
        mode._stop.set()

    mode._run = connect
    mode._thread_main()

    assert attempts == 2
    assert any(state == "reconnecting" for state, _detail in states)
    assert mode._error is None


def test_reconnect_limit_surfaces_terminal_error():
    states = []
    mode = S2SMode(
        S2SConfig(
            reconnect_enabled=True,
            reconnect_attempts=1,
            reconnect_initial_delay=0.001,
            reconnect_max_delay=0.001,
        ),
        on_turn=MagicMock(),
        on_state=lambda state, detail: states.append((state, detail)),
    )
    attempts = 0

    async def fail():
        nonlocal attempts
        attempts += 1
        raise ConnectionError("server unavailable")

    mode._run = fail
    mode._thread_main()

    assert attempts == 2
    assert isinstance(mode._error, ConnectionError)
    assert states[-1][0] == "error"


def test_requirements_probe_reports_reachable_server():
    fake_socket = MagicMock()
    fake_socket.__enter__.return_value = fake_socket
    fake_socket.__exit__.return_value = False
    with patch.dict(sys.modules, {"sounddevice": MagicMock()}), patch(
        "tools.s2s_mode.socket.create_connection", return_value=fake_socket
    ) as connect:
        result = check_s2s_requirements(S2SConfig())

    assert result["available"] is True
    connect.assert_called_once_with(("127.0.0.1", 8765), timeout=2.0)


def test_requirements_rejects_unsupported_audio_rate_without_network_probe():
    with patch.dict(sys.modules, {"sounddevice": MagicMock()}), patch(
        "tools.s2s_mode.socket.create_connection"
    ) as connect:
        result = check_s2s_requirements(S2SConfig(send_rate=22050))

    assert result["available"] is False
    assert "send_rate" in result["details"]
    connect.assert_not_called()
