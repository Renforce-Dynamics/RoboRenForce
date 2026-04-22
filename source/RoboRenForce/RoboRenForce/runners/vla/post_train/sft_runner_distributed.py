"""
VLA SFT Runner (Multi-GPU DDP)

Extends VLASFTRunner with DistributedDataParallel for multi-GPU fine-tuning.

Usage:
    torchrun --nproc_per_node=8 scripts/vla/post_train/train_sft_ddp.py ...
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from RoboRenForce.utils.configclass import configclass
from .sft_runner import VLASFTRunner, VLASFTRunnerCfg


@configclass
class DistributedVLASFTRunnerCfg(VLASFTRunnerCfg):
    """Multi-GPU DDP SFT runner configuration."""

    class_type: type["DistributedVLASFTRunner"] = None

    backend: str = "nccl"
    find_unused_parameters: bool = True


class DistributedVLASFTRunner(VLASFTRunner):
    """Multi-GPU DDP VLA SFT runner.

    Usage:
        torchrun --nproc_per_node=8 train.py
    """

    def __init__(
        self,
        cfg: DistributedVLASFTRunnerCfg,
        log_dir: str = "logs/sft_ddp",
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
            print(f"DDP SFT initialized: {self.world_size} GPUs")

        # 2. Init parent (builds policy, dataset, algorithm, optimizer)
        super().__init__(cfg, log_dir, device)

        # 3. Wrap policy with DDP
        self.policy = DDP(
            self.policy,
            device_ids=[self.local_rank],
            find_unused_parameters=cfg.find_unused_parameters,
        )

        # 4. Replace DataLoaders with distributed versions
        self._setup_distributed_loaders()

    def _setup_distributed_loaders(self):
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

        if self.is_main and self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=0,
            )

    def learn(self, num_epochs: Optional[int] = None):
        num_epochs = num_epochs or self.cfg.num_epochs

        if self.is_main:
            # Unwrap DDP for param counting
            raw = self.policy.module if hasattr(self.policy, "module") else self.policy
            n_trainable = raw.num_trainable_params()
            n_total = raw.num_total_params()
            print(f"Starting DDP SFT: {num_epochs} epochs, "
                  f"{len(self.train_loader)} batches/epoch/GPU, "
                  f"effective batch={self.cfg.batch_size * self.world_size}")
            print(f"  Trainable: {n_trainable:,} / {n_total:,} params "
                  f"({100*n_trainable/max(1,n_total):.1f}%)")

        for epoch in range(num_epochs):
            self.train_sampler.set_epoch(epoch)
            self.policy.train()

            epoch_loss = 0.0
            epoch_steps = 0
            t_start = time.time()

            for batch in self.train_loader:
                batch = self._to_device(batch)

                loss_dict = self.algorithm.update(batch, self.policy, self.optimizer)
                self.scheduler.step()

                epoch_loss += loss_dict["total_loss"]
                epoch_steps += 1
                self.global_step += 1

                if self.is_main and self.global_step % self.cfg.log_interval == 0:
                    self.writer.add_scalar("sft/total_loss", loss_dict["total_loss"], self.global_step)
                    self.writer.add_scalar("sft/lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                    if "kl_loss" in loss_dict:
                        self.writer.add_scalar("sft/kl_loss", loss_dict["kl_loss"], self.global_step)

                if self.is_main and epoch_steps % 100 == 0:
                    elapsed = time.time() - t_start
                    print(f"  step {epoch_steps}/{len(self.train_loader)} | "
                          f"loss={loss_dict['total_loss']:.4f} | "
                          f"time={elapsed:.1f}s", flush=True)

                if (self.is_main and self.cfg.save_interval > 0 and
                        self.global_step % self.cfg.save_interval == 0):
                    self.save_checkpoint()

            # Sync loss
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
                    self.writer.add_scalar("sft/val_loss", val_loss, self.global_step)

            dist.barrier()

        if self.is_main:
            self.save_checkpoint(tag="final")
            print("DDP SFT training complete.")
            if self._writer is not None:
                self._writer.close()

        dist.destroy_process_group()

    def save_checkpoint(self, tag: Optional[str] = None):
        if not self.is_main:
            return

        tag = tag or f"step_{self.global_step}"
        path = Path(self.cfg.checkpoint_dir) / f"sft_checkpoint_{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)

        model = self.policy.module if hasattr(self.policy, "module") else self.policy
        torch.save({
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "policy_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"  Checkpoint saved: {path}")

    @torch.no_grad()
    def validate(self) -> Optional[float]:
        if not self.is_main or not hasattr(self, "val_loader"):
            return None

        self.policy.eval()
        total_loss = 0.0
        n = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            loss_dict = self.algorithm.compute_loss(batch, self.policy)
            total_loss += loss_dict["total_loss"].item()
            n += 1

        self.policy.train()
        return total_loss / max(1, n) if n > 0 else None


DistributedVLASFTRunnerCfg.class_type = DistributedVLASFTRunner
