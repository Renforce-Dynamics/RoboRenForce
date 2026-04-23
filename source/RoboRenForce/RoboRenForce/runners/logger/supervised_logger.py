"""Logger for supervised learning and fine-tuning tasks.

Unlike the RL-specific ``LoggerBase.log()`` which requires runner/env,
this logger provides simple step-based logging for offline training.

Usage:
    logger = SupervisedLogger(cfg, log_dir="results/my_exp")
    logger.init_logger()

    for step in range(total_steps):
        stats = trainer.train_step(batch)
        logger.log_train(step, total_steps, stats, elapsed)

        if step % eval_interval == 0:
            eval_stats = evaluate()
            logger.log_eval(step, eval_stats)

    logger.log_complete(total_steps, total_time)
"""

import time
from typing import Dict, Optional

from RoboRenForce import configclass
from .logger_base import LoggerBase, LoggerBaseCfg


class SupervisedLogger(LoggerBase):
    """Logger for supervised learning / fine-tuning tasks.

    Extends LoggerBase with step-based logging methods that don't
    require an RL environment, reward buffer, or actor-critic.
    """

    def __init__(self, cfg: "SupervisedLoggerCfg", log_dir):
        super().__init__(cfg, log_dir)
        self._start_time: Optional[float] = None
        self._start_step: int = 0

    # ===================================================================== #
    # Public API
    # ===================================================================== #

    def log_train(
        self,
        step: int,
        total_steps: int,
        metrics: Dict[str, float],
        elapsed: Optional[float] = None,
    ):
        """Log a training step to console and writer.

        Args:
            step: Current step number (1-indexed).
            total_steps: Total number of steps.
            metrics: Dict of metric name → value (e.g. {"action_loss": 0.5, "lr": 1e-4}).
            elapsed: Elapsed time in seconds since training start. Auto-tracked if None.
        """
        # Auto-track time
        if self._start_time is None:
            self._start_time = time.time()
            self._start_step = step - 1

        if elapsed is None:
            elapsed = time.time() - self._start_time

        # Log scalars to writer
        self.log_scalars("Train", metrics, step)

        # Console output
        steps_done = step - self._start_step
        sps = steps_done / max(elapsed, 1e-6)
        eta = (total_steps - step) / max(sps, 1e-6)

        parts = [f"step {step}/{total_steps}"]
        for k, v in metrics.items():
            if k == "lr":
                parts.append(f"lr={v:.2e}")
            else:
                parts.append(f"{k}={v:.4f}")
        parts.append(f"{sps:.1f} steps/s")
        parts.append(f"ETA {self._format_time(eta)}")

        print(f"[Train] {' | '.join(parts)}")

    def log_eval(self, step: int, metrics: Dict[str, float]):
        """Log an evaluation result to console and writer.

        Args:
            step: Current step number.
            metrics: Dict of metric name → value.
        """
        self.log_scalars("Eval", metrics, step)

        parts = [f"step {step}"]
        for k, v in metrics.items():
            parts.append(f"{k}={v:.4f}")
        print(f"[Eval]  {' | '.join(parts)}")

    def log_save(self, path: str, step: int):
        """Log a checkpoint save event."""
        print(f"[Save]  {path} (step {step})")

    def log_complete(self, total_steps: int, total_time: float):
        """Log training completion."""
        print(
            f"[Done]  {total_steps} steps in {self._format_time(total_time)}"
        )

    # ===================================================================== #
    # Internals
    # ===================================================================== #

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            m, s = divmod(seconds, 60)
            return f"{int(m)}m{int(s)}s"
        else:
            h, rem = divmod(seconds, 3600)
            m, s = divmod(rem, 60)
            return f"{int(h)}h{int(m)}m"


@configclass
class SupervisedLoggerCfg(LoggerBaseCfg):
    class_type: type[SupervisedLogger] = SupervisedLogger
