"""
VLA Pretraining Runner (Multi-GPU DDP)

Runner for multi-GPU distributed VLA pretraining using PyTorch DDP.
Extends the single-GPU VLAPretrainRunner with distributed data parallel.

Usage:
    torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py ...

Reference: scripts/vla/pretrain/train_ddp.py (standalone DDP script)
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR

from RoboRenForce.utils.configclass import configclass
from .vla_pretrain_runner import VLAPretrainRunner, VLAPretrainRunnerCfg


@configclass
class DistributedVLAPretrainRunnerCfg(VLAPretrainRunnerCfg):
    """Multi-GPU DDP VLA pretraining runner configuration."""

    class_type: type["DistributedVLAPretrainRunner"] = None

    # DDP settings
    backend: str = "nccl"
    find_unused_parameters: bool = True


class DistributedVLAPretrainRunner(VLAPretrainRunner):
    """Multi-GPU DDP VLA pretraining runner.

    Usage:
        torchrun --nproc_per_node=8 train.py

    Env vars set by torchrun: LOCAL_RANK, RANK, WORLD_SIZE
    """

    def __init__(
        self,
        cfg: DistributedVLAPretrainRunnerCfg,
        log_dir: str = "logs/vla_pretrain_ddp",
        device: str = "cuda",
    ):
        # 1. Init distributed
        if not dist.is_initialized():
            dist.init_process_group(backend=cfg.backend)

        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main = (self.rank == 0)
        device = f"cuda:{self.local_rank}"
        torch.cuda.set_device(device)

        if self.is_main:
            print(f"DDP initialized: {self.world_size} GPUs")

        # 2. Init parent (builds model, dataset, algorithm, optimizer)
        super().__init__(cfg, log_dir, device)

        # 3. Wrap model with DDP
        self.vla_actor = DDP(
            self.vla_actor,
            device_ids=[self.local_rank],
            find_unused_parameters=cfg.find_unused_parameters,
        )

        # 4. Replace DataLoaders with distributed versions
        self._setup_distributed_loaders()

    def _setup_distributed_loaders(self):
        """Replace single-GPU DataLoaders with DistributedSampler-based ones."""
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=self.train_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Validation only on rank 0
        if self.is_main and self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=0,
            )

    def learn(self, num_epochs: int = None):
        """DDP training loop with rank-aware logging/saving."""
        num_epochs = num_epochs or self.cfg.num_epochs

        if self.is_main:
            n_params = sum(p.numel() for p in self.vla_actor.parameters() if p.requires_grad)
            print(f"Starting DDP training: {num_epochs} epochs, "
                  f"{len(self.train_loader)} batches/epoch/GPU, "
                  f"{n_params:,} trainable params, "
                  f"effective batch={self.cfg.batch_size * self.world_size}")

        for epoch in range(num_epochs):
            self.train_sampler.set_epoch(epoch)
            self.vla_actor.train()

            epoch_loss = 0.0
            epoch_steps = 0
            t_start = time.time()

            for batch in self.train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                loss_dict = self.algorithm.update(batch, self.vla_actor, self.optimizer)
                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_loss += loss_dict["total_loss"]
                epoch_steps += 1
                self.global_step += 1

                if self.is_main and self.global_step % self.cfg.log_interval == 0:
                    self.writer.add_scalar("train/total_loss", loss_dict["total_loss"], self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

                if self.is_main and epoch_steps % 100 == 0:
                    elapsed = time.time() - t_start
                    print(f"  step {epoch_steps}/{len(self.train_loader)} | "
                          f"loss={loss_dict['total_loss']:.4f} | "
                          f"time={elapsed:.1f}s", flush=True)

                if (self.is_main and self.cfg.save_interval > 0 and
                        self.global_step % self.cfg.save_interval == 0):
                    self.save_checkpoint()

            # Sync average loss across ranks
            avg_loss_t = torch.tensor([epoch_loss / max(1, epoch_steps)], device=self.device)
            dist.all_reduce(avg_loss_t, op=dist.ReduceOp.AVG)

            if self.is_main:
                elapsed = time.time() - t_start
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"loss={avg_loss_t.item():.4f} | "
                      f"steps={epoch_steps} | time={elapsed:.1f}s")

                val_loss = self.validate()
                if val_loss is not None:
                    print(f"  val_loss={val_loss:.4f}")
                    self.writer.add_scalar("val/total_loss", val_loss, self.global_step)

            dist.barrier()

        if self.is_main:
            self.save_checkpoint(tag="final")
            print("DDP training complete.")
            if self._writer is not None:
                self._writer.close()

        dist.destroy_process_group()

    def save_checkpoint(self, tag: str = None):
        """Save checkpoint (rank 0 only), unwrapping DDP module."""
        if not self.is_main:
            return

        tag = tag or f"step_{self.global_step}"
        path = Path(self.cfg.checkpoint_dir) / f"checkpoint_{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Unwrap DDP to get raw module
        model = self.vla_actor.module if hasattr(self.vla_actor, "module") else self.vla_actor

        torch.save({
            "global_step": self.global_step,
            "vla_actor_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint into DDP module."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model = self.vla_actor.module if hasattr(self.vla_actor, "module") else self.vla_actor
        model.load_state_dict(ckpt["vla_actor_state_dict"])

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and self.scheduler and ckpt["scheduler_state_dict"]:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.global_step = ckpt.get("global_step", 0)

        if self.is_main:
            print(f"Checkpoint loaded: step={self.global_step}")

        dist.barrier()

    @torch.no_grad()
    def validate(self) -> Optional[float]:
        """Validation (rank 0 only)."""
        if not self.is_main or not hasattr(self, "val_loader"):
            return None

        self.vla_actor.eval()
        total_loss = 0.0
        n = 0

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            loss_dict = self.algorithm.compute_loss(batch, self.vla_actor)
            total_loss += loss_dict["total_loss"].item()
            n += 1

        self.vla_actor.train()
        return total_loss / max(1, n) if n > 0 else None


# Set class_type after class definition
DistributedVLAPretrainRunnerCfg.class_type = DistributedVLAPretrainRunner
