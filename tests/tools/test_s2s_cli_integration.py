"""CLI routing tests for voice-originated realtime S2S turns."""

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI
from tools.s2s_mode import S2STurn


def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli._s2s_mode = None
    cli._s2s_state = "off"
    cli._s2s_partial = ""
    cli._s2s_last_error = ""
    cli._voice_mode = False
    cli._pending_input = queue.Queue()
    cli._agent_running = False
    cli.agent = None
    cli._approval_state = None
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._sudo_state = None
    cli._secret_state = None
    cli._app = None
    cli._last_invalidate = 0.0
    cli._invalidate = MagicMock()
    return cli


class _FakeMode:
    is_running = True

    def __init__(self):
        self.responses = []
        self.stopped = False

    def deliver_response(self, call_id, response):
        self.responses.append((call_id, response))

    def stop(self):
        self.stopped = True


def test_voice_turn_enters_normal_cli_input_queue():
    cli = _make_cli()
    turn = S2STurn("inspect the current repository", "call-1")

    ack = cli._on_s2s_turn(turn)

    assert cli._pending_input.get_nowait() == turn
    assert ack.await_result is True
    assert "working in the terminal" in ack.message
    cli._invalidate.assert_called()


def test_explicit_voice_approval_resolves_active_prompt():
    cli = _make_cli()
    mode = _FakeMode()
    cli._s2s_mode = mode
    approval_queue = queue.Queue()
    cli._approval_state = {
        "choices": ["once", "session", "deny"],
        "response_queue": approval_queue,
    }

    ack = cli._on_s2s_turn(S2STurn("approve once", "call-2"))

    assert approval_queue.get_nowait() == "once"
    assert cli._approval_state is None
    assert ack.message == "Approved the command once."
    assert ack.await_result is False
    assert mode.responses == []


def test_ambiguous_voice_approval_is_not_guessed():
    cli = _make_cli()
    mode = _FakeMode()
    cli._s2s_mode = mode
    approval_queue = queue.Queue()
    cli._approval_state = {
        "choices": ["once", "deny"],
        "response_queue": approval_queue,
    }

    ack = cli._on_s2s_turn(S2STurn("maybe", "call-3"))

    assert approval_queue.empty()
    assert cli._approval_state is not None
    assert "approve once or deny" in ack.message
    assert ack.await_result is False


def test_exact_stop_phrase_interrupts_active_agent():
    cli = _make_cli()
    mode = _FakeMode()
    cli._s2s_mode = mode
    cli._agent_running = True
    cli.agent = SimpleNamespace(interrupt=MagicMock())

    ack = cli._on_s2s_turn(S2STurn("stop it", "call-4"))

    cli.agent.interrupt.assert_called_once_with()
    assert cli._pending_input.empty()
    assert ack.message == "Stopping Hermes now."
    assert ack.await_result is False
    assert mode.responses == []


def test_completed_cli_turn_is_returned_to_s2s_mode():
    cli = _make_cli()
    mode = _FakeMode()
    cli._s2s_mode = mode

    cli._complete_s2s_turn("call-5", "Hermes finished")

    assert mode.responses == [("call-5", "Hermes finished")]


def test_s2s_command_routes_to_mode_lifecycle():
    cli = _make_cli()
    cli._enable_s2s_mode = MagicMock()
    cli._disable_s2s_mode = MagicMock()
    cli._show_s2s_status = MagicMock()

    cli._handle_s2s_command("/s2s on")
    cli._handle_s2s_command("/s2s off")
    cli._handle_s2s_command("/s2s status")

    cli._enable_s2s_mode.assert_called_once_with()
    cli._disable_s2s_mode.assert_called_once_with()
    cli._show_s2s_status.assert_called_once_with()


@patch("cli._cprint")
def test_unknown_s2s_command_prints_usage(mock_print):
    cli = _make_cli()

    cli._handle_s2s_command("/s2s unexpected")

    assert any("Usage: /s2s" in str(call) for call in mock_print.call_args_list)
