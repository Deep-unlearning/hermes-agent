"""CLI routing tests for voice-originated realtime S2S turns."""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI
from tools.s2s_mode import S2SControlCall, S2STurn


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
    cli._background_tasks = {}
    cli._background_task_info = {}
    cli._background_task_lock = threading.RLock()
    cli._background_task_counter = 0
    cli._prompt_start_time = None
    cli._spinner_text = ""
    cli._tool_start_time = 0.0
    cli._approval_state = None
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._sudo_state = None
    cli._secret_state = None
    cli._app = None
    cli._last_invalidate = 0.0
    cli._invalidate = MagicMock()
    cli._clear_active_overlays_for_interrupt = MagicMock()
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


def test_status_control_reports_live_tool_activity_and_queue():
    cli = _make_cli()
    cli._agent_running = True
    cli._prompt_start_time = time.time() - 65
    cli._spinner_text = "running pytest"
    cli._tool_start_time = time.monotonic()
    cli._pending_input.put("next request")

    ack = cli._on_s2s_control(
        S2SControlCall("get_hermes_status", "status-1")
    )

    assert ack.await_result is False
    assert "1 minute 5 seconds" in ack.message
    assert "running pytest" in ack.message
    assert "1 request is queued next" in ack.message


def test_steer_control_updates_active_agent_without_queueing_new_turn():
    cli = _make_cli()
    cli._agent_running = True
    cli.agent = SimpleNamespace(steer=MagicMock(return_value=True))

    ack = cli._on_s2s_control(
        S2SControlCall("steer_hermes", "steer-1", "focus on the failing test")
    )

    cli.agent.steer.assert_called_once_with("focus on the failing test")
    assert cli._pending_input.empty()
    assert ack.await_result is False
    assert "steered" in ack.message


def test_queue_control_defers_a_voice_task_and_preserves_result_identity():
    cli = _make_cli()
    call = S2SControlCall(
        "queue_hermes_task", "queue-1", message="run the integration tests later"
    )

    ack = cli._on_s2s_control(call)

    queued = cli._pending_input.get_nowait()
    assert queued == S2STurn("run the integration tests later", "queue-1")
    assert ack.await_result is True
    assert "queued" in ack.message


def test_stop_control_interrupts_active_agent_and_clears_prompts():
    cli = _make_cli()
    cli._agent_running = True
    cli.agent = SimpleNamespace(interrupt=MagicMock())

    ack = cli._on_s2s_control(S2SControlCall("stop_hermes", "stop-1"))

    cli.agent.interrupt.assert_called_once_with()
    cli._clear_active_overlays_for_interrupt.assert_called_once_with()
    assert ack.await_result is False
    assert "Stopping" in ack.message


def test_background_control_uses_btw_and_announces_completion():
    cli = _make_cli()
    mode = _FakeMode()
    cli._s2s_mode = mode
    cli._handle_background_command = MagicMock(return_value="bg-task-1")
    call = S2SControlCall(
        "start_background_task", "background-1", "inspect the test failures"
    )

    ack = cli._on_s2s_control(call)

    command = cli._handle_background_command.call_args.args[0]
    completion_callback = cli._handle_background_command.call_args.kwargs[
        "completion_callback"
    ]
    completion_callback("Tests are passing.")
    assert command == "/btw inspect the test failures"
    assert ack.await_result is True
    assert mode.responses == [("background-1", "Tests are passing.")]


def test_background_controls_list_and_stop_one_task_by_spoken_number():
    cli = _make_cli()
    stop_event = threading.Event()
    background_agent = SimpleNamespace(
        interrupt=MagicMock(), steer=MagicMock(return_value=True)
    )
    cli._background_task_info["bg_123"] = {
        "task_id": "bg_123",
        "number": 1,
        "prompt": "inspect the logs",
        "status": "running",
        "activity": "reading server.log",
        "response": "",
        "started_at": time.time() - 12,
        "finished_at": None,
        "agent": background_agent,
        "stop_event": stop_event,
    }

    status = cli._on_s2s_control(
        S2SControlCall("get_background_tasks", "list-1")
    )
    steered = cli._on_s2s_control(
        S2SControlCall(
            "steer_background_task",
            "steer-bg-1",
            message="focus on errors",
            task_id="one",
        )
    )
    stopped = cli._on_s2s_control(
        S2SControlCall("stop_background_task", "stop-bg-1", task_id="one")
    )

    assert "Task 1" in status.message
    assert "reading server.log" in status.message
    assert "Steered background task 1" in steered.message
    assert "Stopping background task 1" in stopped.message
    assert stop_event.is_set()
    background_agent.interrupt.assert_called_once_with()
    background_agent.steer.assert_called_once_with("focus on errors")
    assert cli._background_task_info["bg_123"]["status"] == "stopping"


def test_progress_provider_reports_background_activity_when_foreground_is_idle():
    cli = _make_cli()
    cli._background_task_info["bg_456"] = {
        "task_id": "bg_456",
        "number": 2,
        "prompt": "run integration tests",
        "status": "running",
        "activity": "executing pytest",
        "response": "",
        "started_at": time.time() - 20,
        "finished_at": None,
        "agent": None,
        "stop_event": threading.Event(),
    }

    progress = cli._get_s2s_progress_announcement()

    assert progress is not None
    assert "Background progress" in progress
    assert "executing pytest" in progress


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
