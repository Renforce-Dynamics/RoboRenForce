"""Minimal logger for AI-agent / LLM consumers.

Writes append-only ``events.jsonl`` and a small ``summary.md`` snapshot under
``log_dir``, plus single-line stdout. Optional TB/W&B writer via
``cfg.enable_writer=True``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from RoboRenForce import configclass
from .logger_base import LoggerBase, LoggerBaseCfg


_SUMMARY_KINDS = ("train", "eval", "save", "done")


class AgentLogger(LoggerBase):
    """Append-only structured logger designed for an LLM to grep / tail."""

    def __init__(self, cfg: "AgentLoggerCfg", log_dir):
        super().__init__(cfg, log_dir)
        self._start_time: Optional[float] = None
        self._start_step: Optional[int] = None
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._events_path: Optional[Path] = None
        self._events_fh = None
        self._summary_path: Optional[Path] = None
        self._summary_last_write: float = 0.0

    def init_logger(self):
        if self.log_dir is not None:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            self._events_path = Path(self.log_dir) / self.cfg.events_filename
            self._summary_path = Path(self.log_dir) / self.cfg.summary_filename
            self._events_fh = self._events_path.open("a", buffering=1)
        if self.cfg.enable_writer:
            super().init_logger()
        else:
            # Set logger_type so save_model() (which inspects it) does not crash.
            self.logger_type = "none"
            print(f"[agent logger] writing to {self.log_dir}")

    def close(self):
        if self._events_fh is not None:
            self._events_fh.close()
            self._events_fh = None
        if self._latest and self._summary_path is not None:
            self._write_summary()

    def log_train(
        self,
        step: int,
        total_steps: int,
        metrics: Dict[str, float],
        elapsed: Optional[float] = None,
    ):
        first_call = self._start_time is None
        if first_call:
            self._start_time = time.time()
            self._start_step = step
        if elapsed is None:
            elapsed = time.time() - self._start_time

        clean = self._scalarize(metrics)
        self.log_scalars("Train", clean, step)

        event: Dict[str, Any] = {
            "t": round(time.time(), 3),
            "kind": "train",
            "step": step,
            "total_steps": total_steps,
            "metrics": clean,
            "wall_s": round(elapsed, 3),
        }
        if not first_call and elapsed > 0:
            steps_done = step - self._start_step
            sps = steps_done / elapsed
            event["sps"] = round(sps, 3)
            event["eta_s"] = round((total_steps - step) / max(sps, 1e-9), 1)
        self._emit(event, force_summary=False)

        parts = [f"step={step}/{total_steps}"]
        for k, v in clean.items():
            parts.append(f"{k}={v:.2e}" if k == "lr" else f"{k}={v:.4f}")
        if "sps" in event:
            parts.append(f"sps={event['sps']:.1f}")
            parts.append(f"eta={self._format_time(event['eta_s'])}")
        print("[agent train] " + " ".join(parts), flush=True)

    def log_eval(self, step: int, metrics: Dict[str, float]):
        clean = self._scalarize(metrics)
        self.log_scalars("Eval", clean, step)
        self._emit(
            {"t": round(time.time(), 3), "kind": "eval", "step": step, "metrics": clean},
            force_summary=True,
        )
        parts = [f"step={step}"] + [f"{k}={v:.4f}" for k, v in clean.items()]
        print("[agent eval]  " + " ".join(parts), flush=True)

    def log_save(self, path: str, step: int):
        self._emit(
            {"t": round(time.time(), 3), "kind": "save", "step": step, "path": str(path)},
            force_summary=True,
        )
        print(f"[agent save]  step={step} path={path}", flush=True)

    def log_complete(self, total_steps: int, total_time: float):
        self._emit(
            {
                "t": round(time.time(), 3),
                "kind": "done",
                "total_steps": total_steps,
                "total_time_s": round(total_time, 1),
            },
            force_summary=True,
        )
        print(f"[agent done]  steps={total_steps} time={self._format_time(total_time)}", flush=True)
        self.close()

    def _scalarize(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in metrics.items():
            f = self._coerce_scalar(v)
            if f is not None:
                out[k] = f
        return out

    def _emit(self, event: Dict[str, Any], force_summary: bool):
        if self._events_fh is not None:
            self._events_fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._latest[event["kind"]] = event
        if self._summary_path is None:
            return
        now = time.monotonic()
        if force_summary or (now - self._summary_last_write) >= self.cfg.summary_min_interval_s:
            self._write_summary()
            self._summary_last_write = now

    def _write_summary(self):
        lines = ["# Run summary", ""]
        for kind in _SUMMARY_KINDS:
            event = self._latest.get(kind)
            if not event:
                continue
            lines.append(f"## latest {kind}")
            for k, v in event.items():
                if k == "metrics":
                    for mk, mv in v.items():
                        lines.append(f"- **{kind}.{mk}**: {mv}")
                else:
                    lines.append(f"- **{k}**: {v}")
            lines.append("")
        self._summary_path.write_text("\n".join(lines))


@configclass
class AgentLoggerCfg(LoggerBaseCfg):
    class_type: type[AgentLogger] = AgentLogger

    enable_writer: bool = False
    """If True, also init the parent TB/Wandb/Neptune writer for graphs."""

    events_filename: str = "events.jsonl"
    summary_filename: str = "summary.md"

    summary_min_interval_s: float = 1.0
    """Minimum seconds between summary.md rewrites for non-eval/save/done events."""
