"""Track isolated child process groups and clean them up on parent exit."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any


class ProcessGroupRegistry:
    """Own process groups started by one bounded command-line tool."""

    def __init__(self) -> None:
        self.active: dict[int, subprocess.Popen[Any]] = {}
        self._termination_signals = (signal.SIGTERM, signal.SIGINT)
        self._spawning = False
        self._pending_signal: int | None = None

    def register(self, process: subprocess.Popen[Any]) -> None:
        self.active[process.pid] = process

    def unregister(self, process: subprocess.Popen[Any]) -> None:
        self.active.pop(process.pid, None)

    def spawn(self, *args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        """Start and register a child without a parent-signal orphan window."""

        if self._spawning:
            raise RuntimeError("nested managed process creation is unsupported")
        self._spawning = True
        try:
            process = subprocess.Popen(*args, **kwargs)
            self.register(process)
        finally:
            self._spawning = False
            if self._pending_signal is not None:
                signum = self._pending_signal
                self._pending_signal = None
                self.kill_all()
                raise SystemExit(128 + signum)
        return process

    def kill_all(self) -> None:
        """Kill and reap every tracked process group."""

        processes = list(self.active.values())
        for process in processes:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        for process in processes:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1.0)
            if process.poll() is not None:
                self.unregister(process)

    def _termination_handler(self, signum: int, _frame: object) -> None:
        if self._spawning:
            self._pending_signal = signum
            return
        self.kill_all()
        raise SystemExit(128 + signum)

    @contextmanager
    def cleanup_on_parent_exit(self) -> Iterator[None]:
        """Kill tracked groups on SIGTERM, SIGINT, or ordinary stack unwinding."""

        if threading.current_thread() is not threading.main_thread():
            yield
            return
        previous = {value: signal.getsignal(value) for value in self._termination_signals}
        for value in self._termination_signals:
            signal.signal(value, self._termination_handler)
        try:
            yield
        finally:
            self.kill_all()
            for value, handler in previous.items():
                signal.signal(value, handler)
