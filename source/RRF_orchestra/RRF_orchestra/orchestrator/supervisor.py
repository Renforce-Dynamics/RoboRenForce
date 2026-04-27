"""Supervisor — owns the spawned processes, drives lifecycle.

Use as a context manager:

    with Supervisor(topology_cfg) as sup:
        sup.start_all()
        for it in range(max_iter):
            trajs = sup.collect_trajectories(min_count=batch_size)
            ...
        # __exit__ broadcasts stop and joins all processes

A daemon thread inside the Supervisor watches `ctrl_out_ch` for `fatal_error`
messages from any worker and flips an internal stop flag — the next call to
`should_stop()` returns True so the learner loop bails out.
"""

from __future__ import annotations

import os
import queue
import signal
import threading
import time
import traceback
from typing import Optional

import torch.multiprocessing as mp

from RRF_orchestra.orchestrator.topology import (
    BuiltTopology,
    Topology,
    TopologyCfg,
)
from RRF_orchestra.protocol.messages import (
    ControlMsg,
    Trajectory,
    WeightUpdate,
)

CTX = mp.get_context("spawn")


class TopologyError(RuntimeError):
    """Raised when a worker reports `fatal_error` or fails to start."""


class Supervisor:
    """Spawn-and-supervise the topology declared by `TopologyCfg`."""

    def __init__(self, cfg: TopologyCfg):
        self.cfg = cfg
        self.built: Optional[BuiltTopology] = None
        self._procs: list[mp.Process] = []
        self._proc_names: list[str] = []
        self._stop_event = threading.Event()
        self._fatal: Optional[ControlMsg] = None
        self._watcher: Optional[threading.Thread] = None
        self._installed_sigint: bool = False
        self._prev_sigint = None

    # ---- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "Supervisor":
        # Build (but do NOT spawn) so that callers can introspect channels.
        self.built = Topology(self.cfg).build()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.stop_all()
        finally:
            self._restore_sigint()

    def start_all(self) -> None:
        """Spawn one process per env worker + 1 inference worker."""
        if self.built is None:
            raise RuntimeError("Supervisor.start_all called outside of context manager")
        if self._procs:
            raise RuntimeError("start_all called twice")

        # Install a SIGINT handler that flips the stop flag (so Ctrl-C in the
        # learner shuts the topology down cleanly instead of leaving zombies).
        self._install_sigint()

        for ew in self.built.env_workers:
            p = CTX.Process(target=ew.run, name=ew.cfg.name)
            p.start()
            self._procs.append(p)
            self._proc_names.append(ew.cfg.name)

        ip = CTX.Process(
            target=self.built.inference_worker.run,
            name=self.built.inference_worker.cfg.name,
        )
        ip.start()
        self._procs.append(ip)
        self._proc_names.append(self.built.inference_worker.cfg.name)

        self._watcher = threading.Thread(
            target=self._watch_ctrl_out, name="supervisor-watch", daemon=True
        )
        self._watcher.start()

    def stop_all(self, join_timeout_s: float = 10.0) -> None:
        """Send `stop` to every worker, join, then SIGTERM stragglers."""
        if not self._procs:
            return

        self._stop_event.set()

        if self.built is not None:
            for ch in self.built.ctrl_in_chs:
                try:
                    ch.put(
                        ControlMsg(kind="stop", sender="supervisor", timestamp=time.time()),
                        timeout=1.0,
                    )
                except Exception:
                    pass

        for p in self._procs:
            p.join(timeout=join_timeout_s)
        for p in self._procs:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    traceback.print_exc()
        for p in self._procs:
            p.join(timeout=2.0)

        self._procs.clear()
        self._proc_names.clear()

    # ---- learner-side helpers ---------------------------------------------

    def should_stop(self) -> bool:
        """True if either a fatal error or an explicit stop has been raised."""
        return self._stop_event.is_set()

    def fatal(self) -> Optional[ControlMsg]:
        """The first fatal_error message seen (None if topology healthy)."""
        return self._fatal

    def collect_trajectories(
        self, min_count: int, timeout_s: float = 60.0
    ) -> list[Trajectory]:
        """Block until ≥ min_count trajectories are available, or timeout/stop."""
        if self.built is None or self.built.traj_ch is None:
            raise RuntimeError(
                "collect_trajectories requires TopologyCfg.enable_traj_channel=True"
            )
        deadline = time.monotonic() + timeout_s
        out: list[Trajectory] = []
        while len(out) < min_count and not self.should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                t = self.built.traj_ch.get(timeout=min(0.5, max(0.01, remaining)))
            except Exception:
                continue
            out.append(t)
        if self._fatal is not None:
            raise TopologyError(
                f"worker fatal_error: {self._fatal.sender}: "
                f"{self._fatal.payload}"
            )
        return out

    def broadcast_weights(self, msg: WeightUpdate) -> None:
        """Push a `WeightUpdate` to the inference worker. Drops oldest first."""
        if self.built is None or self.built.weight_ch is None:
            raise RuntimeError(
                "broadcast_weights requires TopologyCfg.enable_weight_channel=True"
            )
        # Drain any pending weight updates so newest replaces oldest.
        while True:
            try:
                _ = self.built.weight_ch.get(timeout=0)
            except Exception:
                break
        try:
            self.built.weight_ch.put(msg, timeout=1.0)
        except Exception as exc:
            raise TopologyError(f"weight broadcast failed: {exc!r}") from exc

    # ---- internals ---------------------------------------------------------

    def _watch_ctrl_out(self) -> None:
        if self.built is None:
            return
        ch = self.built.ctrl_out_ch
        while not self._stop_event.is_set():
            try:
                msg = ch.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception:
                continue
            if not isinstance(msg, ControlMsg):
                continue
            if msg.kind == "fatal_error":
                self._fatal = msg
                self._stop_event.set()
                return
            # Other kinds (health_pong, etc.) are ignored in v1.

    def _install_sigint(self) -> None:
        if self._installed_sigint:
            return
        try:
            self._prev_sigint = signal.signal(signal.SIGINT, self._on_sigint)
            self._installed_sigint = True
        except ValueError:
            # Not on the main thread — Supervisor is being used inside a thread
            # already. Skip handler install; caller is responsible for stop.
            self._installed_sigint = False

    def _on_sigint(self, signum, frame):
        self._stop_event.set()
        # Restore previous handler so a second Ctrl-C kills hard.
        try:
            signal.signal(signal.SIGINT, self._prev_sigint or signal.SIG_DFL)
        except Exception:
            pass

    def _restore_sigint(self) -> None:
        if not self._installed_sigint:
            return
        try:
            signal.signal(signal.SIGINT, self._prev_sigint or signal.SIG_DFL)
        except Exception:
            pass
        self._installed_sigint = False
