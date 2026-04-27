"""BaseWorker — common lifecycle for processes spawned by the Supervisor.

Each subclass overrides `setup()`, `run_loop()`, `teardown()`. The provided
`run()` wraps them with:
  - SIGTERM/SIGINT handlers that flip an internal stop flag,
  - a daemon thread that watches `ctrl_ch_in` for `ControlMsg(kind="stop")`,
  - exception capture that emits a `fatal_error` ControlMsg to `ctrl_ch_out`
    so the Supervisor can shut down the rest of the topology.

The worker object itself is constructed in the parent and pickled across the
spawn boundary, so `__init__` MUST stay cheap (no torch loads, no sim init).
Heavy initialization belongs in `setup()`, which runs inside the child.
"""

from __future__ import annotations

import os
import signal
import threading
import time
import traceback
from dataclasses import field
from typing import Optional

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template import ClassTemplateBase, ClassTemplateBaseCfg

from RRF_orchestra.protocol.channels import Channel
from RRF_orchestra.protocol.messages import ControlMsg


@configclass
class BaseWorkerCfg(ClassTemplateBaseCfg):
    """Fields shared by every worker. Subclass adds its own."""

    name: str = "worker"
    # Wired by the Supervisor / Topology before the worker is spawned. They are
    # plain Channel instances (which wrap a spawn-context mp.Queue) so they
    # pickle cleanly across the spawn boundary.
    ctrl_ch_in: Optional[Channel] = None    # Supervisor → this worker
    ctrl_ch_out: Optional[Channel] = None   # this worker → Supervisor
    # How often the loop checks for a stop flag when otherwise idle.
    poll_interval_s: float = 0.05


class BaseWorker(ClassTemplateBase):
    """Abstract worker process body.

    Subclasses must implement `setup`, `run_loop`, `teardown`. Do NOT override
    `run` unless you understand the lifecycle plumbing.
    """

    cfg: BaseWorkerCfg

    def __init__(self, cfg: BaseWorkerCfg):
        self.cfg = cfg
        # Created lazily inside `run()` so the un-pickled child has fresh state.
        self._stop_event: Optional[threading.Event] = None
        self._ctrl_thread: Optional[threading.Thread] = None

    # ---- subclass hooks ----------------------------------------------------

    def setup(self) -> None:
        """Heavy init inside the child process (sim engines, model load, ...)."""

    def run_loop(self) -> None:
        """Main work loop. Must check `self.should_stop()` and return when set."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Release resources. Always called, even on exception."""

    # ---- provided ----------------------------------------------------------

    def should_stop(self) -> bool:
        return self._stop_event is not None and self._stop_event.is_set()

    def request_stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    def run(self) -> None:
        """Process entry point. Pickled with `self` and invoked by mp.Process."""
        self._stop_event = threading.Event()
        self._install_signal_handlers()
        self._start_ctrl_watcher()
        try:
            self.setup()
            self.run_loop()
        except KeyboardInterrupt:
            # Clean shutdown path on Ctrl-C.
            pass
        except BaseException as exc:  # noqa: BLE001 — must surface to supervisor
            self._emit_fatal(exc)
            raise
        finally:
            self.request_stop()
            try:
                self.teardown()
            except BaseException:  # noqa: BLE001
                # Don't mask the original error; just log to stderr.
                traceback.print_exc()

    # ---- internal ----------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        def _on_signal(signum, _frame):
            # Flag-only handler; the loop checks should_stop() at its own pace.
            self.request_stop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _on_signal)
            except ValueError:
                # Not on the main thread of the process — give up silently.
                pass

    def _start_ctrl_watcher(self) -> None:
        if self.cfg.ctrl_ch_in is None:
            return
        ch = self.cfg.ctrl_ch_in

        def _watch():
            while not self.should_stop():
                try:
                    msg = ch.get(timeout=self.cfg.poll_interval_s)
                except Exception:
                    continue
                if isinstance(msg, ControlMsg) and msg.kind == "stop":
                    self.request_stop()
                    return

        t = threading.Thread(target=_watch, name=f"{self.cfg.name}-ctrl", daemon=True)
        t.start()
        self._ctrl_thread = t

    def _emit_fatal(self, exc: BaseException) -> None:
        if self.cfg.ctrl_ch_out is None:
            return
        try:
            self.cfg.ctrl_ch_out.put(
                ControlMsg(
                    kind="fatal_error",
                    sender=self.cfg.name,
                    payload={
                        "pid": os.getpid(),
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    timestamp=time.time(),
                ),
                timeout=1.0,
            )
        except Exception:
            # Supervisor channel may be closed already; nothing actionable.
            pass
