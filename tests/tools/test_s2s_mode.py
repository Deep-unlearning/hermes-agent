"""Behavior tests for the native CLI realtime S2S transport."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.s2s_mode import (
    S2SConfig,
    S2SMode,
    S2STurn,
    S2STurnAck,
    build_session_update,
    check_s2s_requirements,
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
        }
    )

    assert config.host == "localhost"
    assert config.port == 8765
    assert config.voice == "af_heart"
    assert config.send_rate == 16000
    assert config.recv_rate == 24000
    assert config.input_device is None
    assert config.block_mic_during_playback is True


def test_session_update_exposes_only_scoped_hermes_tool():
    event = build_session_update(S2SConfig(voice="af_heart", recv_rate=24000))
    session = event["session"]

    assert event["type"] == "session.update"
    assert [tool["name"] for tool in session["tools"]] == ["send_to_hermes"]
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


def test_deliver_response_truncates_at_configured_spoken_limit():
    mode = S2SMode(S2SConfig(max_spoken_chars=8), on_turn=MagicMock())
    mode.deliver_response("call-1", "abcdefghijk")

    outbound = mode._responses.get_nowait()
    assert outbound.kind == "hermes_result"
    assert outbound.call_id == "call-1"
    assert outbound.text == "abcdefgh …(truncated)"


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
