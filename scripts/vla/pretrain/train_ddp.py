#!/usr/bin/env python3
"""
VLA Pretrain — Multi-GPU DDP Training Script

Usage:
    # 2 GPUs with Psi0 data (mock VLM):
    torchrun --nproc_per_node=2 scripts/vla/pretrain/train_ddp.py \
        --psi0 --epochs 5 --batch_size 16

    # 8 GPUs:
    torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
        --psi0 --epochs 10 --batch_size 32 --amp

    # With real Qwen2-VL:
    torchrun --nproc_per_node=8 scripts/vla/pretrain/train_ddp.py \
        --data_root /path/to/data --epochs 20 --amp
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../source/RoboRenForce"))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.optim.lr_scheduler import LambdaLR

from RoboRenForce.utils.configclass import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.vlm.vlm_backbone_base import VLMBackbone, VLMBackboneCfg
from RoboRenForce.networks.vlm.fusion_layers import FusionLayerCfg
from RoboRenForce.components.actor.action_heads.regression_action_head import RegressionActionHeadCfg
from RoboRenForce.components.actor.action_heads.diffusion_action_head import DiffusionActionHeadCfg
from RoboRenForce.components.actor.vla_actor import VLAActorCfg
from RoboRenForce.algorithms.vla_training.pretrain_algorithm import VLAPretrainAlgorithmCfg
from RoboRenForce.prototype.embodied.lerobot.lerobot_dataset import LeRobotDatasetCfg


# ===== Mock VLM ===== #

class MockVLM(VLMBackbone):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, cfg.output_dim),
        )
        self.output_dim = cfg.output_dim

    def forward(self, image, text=None, **kwargs):
        return self.encoder(image)


@configclass
class MockVLMCfg(VLMBackboneCfg):
    class_type: type = MockVLM
    model_name: str = "mock"
    output_dim: int = 256
    freeze: bool = True


class _MockDataset(torch.utils.data.Dataset):
    def __init__(self, action_dim=36, n=500):
        self.n = n
        self.action_dim = action_dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 64, 64),
            "proprioception": torch.randn(32),
            "action": torch.randn(self.action_dim),
        }


# ===== DDP Runner ===== #

class VLAPretrainDDPRunner:
    """
    Multi-GPU DDP training runner for VLA pretraining.

    Key differences from single-GPU runner:
    - Uses DistributedDataParallel for model wrapping
    - Uses DistributedSampler for data loading
    - Only rank 0 does logging, checkpointing, validation
    - Gradient sync handled automatically by DDP
    """

    def __init__(self, args):
        # DDP setup
        dist.init_process_group(backend="nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        self.args = args
        self.is_main = (self.rank == 0)

        if self.is_main:
            print(f"DDP initialized: {self.world_size} GPUs")
            print(f"  Action head: {args.head}")
            print(f"  Batch size per GPU: {args.batch_size}")
            print(f"  Effective batch size: {args.batch_size * self.world_size}")

        # 1. Dataset
        if args.mock:
            self.dataset = _MockDataset(action_dim=args.action_dim)
        else:
            dataset_cfg = LeRobotDatasetCfg(
                data_root=args.data_root,
                load_videos=True,
                frames_dir=args.frames_dir,
                image_size=tuple(args.image_size),
            )
            self.dataset = dataset_cfg.construct_from_cfg()

        n_total = len(self.dataset)
        n_val = max(1, int(n_total * 0.05))
        n_train = n_total - n_val
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_sampler = DistributedSampler(
            self.train_dataset, num_replicas=self.world_size,
            rank=self.rank, shuffle=True,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            sampler=self.train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if self.is_main:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # 2. Model
        dim_params = self._get_dim_params()

        if args.psi0 or args.mock:
            vlm_cfg = MockVLMCfg(output_dim=256)
        else:
            from RoboRenForce.networks.vlm.qwen2vl import Qwen2VLCfg
            vlm_cfg = Qwen2VLCfg(model_name=args.model_name, freeze=True)

        action_head_cfg = RegressionActionHeadCfg(
            action_dim=args.action_dim, action_horizon=1, hidden_dims=[256, 256],
        ) if args.head == "regression" else DiffusionActionHeadCfg(
            action_dim=args.action_dim, action_horizon=1,
            num_layers=4, num_heads=8, embed_dim=256,
            num_train_steps=100, num_diffusion_steps=10,
        )

        actor_cfg = VLAActorCfg(
            vlm_backbone_cfg=vlm_cfg,
            freeze_vlm=True,
            fusion_cfg=FusionLayerCfg(output_dim=512, hidden_dims=[512]),
            action_head_cfg=action_head_cfg,
            use_proprioception=True,
        )

        self.vla_actor = actor_cfg.construct_from_cfg(dim_params)
        self.vla_actor.to(self.device)

        # Wrap with DDP
        self.vla_actor = DDP(
            self.vla_actor,
            device_ids=[self.local_rank],
            find_unused_parameters=True,
        )

        # 3. Algorithm
        algo_cfg = VLAPretrainAlgorithmCfg(
            learning_rate=args.lr, warmup_steps=args.warmup_steps,
            use_amp=args.amp,
        )
        self.algorithm = algo_cfg.construct_from_cfg()

        # 4. Optimizer
        trainable_params = [p for p in self.vla_actor.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=1e-4,
        )

        # 5. Scheduler
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / max(1, args.warmup_steps)),
        )

        self.global_step = 0
        self.checkpoint_dir = Path(args.checkpoint_dir)
        if self.is_main:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._writer = None

    @property
    def writer(self):
        if self._writer is None and self.is_main:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=self.args.log_dir)
            except ImportError:
                class _Dummy:
                    def add_scalar(self, *a, **kw): pass
                    def close(self): pass
                self._writer = _Dummy()
        return self._writer

    def learn(self):
        num_epochs = self.args.epochs
        if self.is_main:
            n_params = sum(p.numel() for p in self.vla_actor.parameters() if p.requires_grad)
            print(f"Starting DDP training: {num_epochs} epochs, "
                  f"{len(self.train_loader)} batches/epoch/GPU, "
                  f"{n_params:,} trainable params")

        for epoch in range(num_epochs):
            self.train_sampler.set_epoch(epoch)
            self.vla_actor.train()

            epoch_loss = 0.0
            epoch_steps = 0
            t_start = time.time()

            for batch in self.train_loader:
                batch = self._to_device(batch)

                loss_dict = self.algorithm.update(batch, self.vla_actor, self.optimizer)
                self.scheduler.step()

                epoch_loss += loss_dict["total_loss"]
                epoch_steps += 1
                self.global_step += 1

                if self.is_main and self.global_step % 10 == 0:
                    self.writer.add_scalar("train/total_loss", loss_dict["total_loss"], self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

                if self.is_main and epoch_steps % 100 == 0:
                    elapsed_so_far = time.time() - t_start
                    print(f"  step {epoch_steps}/{len(self.train_loader)} | "
                          f"loss={loss_dict['total_loss']:.4f} | "
                          f"time={elapsed_so_far:.1f}s", flush=True)

                if self.is_main and self.args.save_interval > 0 and self.global_step % self.args.save_interval == 0:
                    self._save_checkpoint()

            # Sync loss across ranks
            avg_loss_tensor = torch.tensor([epoch_loss / max(1, epoch_steps)], device=self.device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)

            if self.is_main:
                elapsed = time.time() - t_start
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"loss={avg_loss_tensor.item():.4f} | "
                      f"steps={epoch_steps} | time={elapsed:.1f}s")

                val_loss = self._validate()
                self.writer.add_scalar("val/total_loss", val_loss, self.global_step)
                print(f"  val_loss={val_loss:.4f}")

            dist.barrier()

        if self.is_main:
            self._save_checkpoint(tag="final")
            print("DDP training complete.")
            if self.writer:
                self.writer.close()

        dist.destroy_process_group()

    @torch.no_grad()
    def _validate(self) -> float:
        self.vla_actor.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            loss_dict = self.algorithm.compute_loss(batch, self.vla_actor)
            total_loss += loss_dict["total_loss"].item()
            n_batches += 1

        self.vla_actor.train()
        return total_loss / max(1, n_batches)

    def _save_checkpoint(self, tag=None):
        tag = tag or f"step_{self.global_step}"
        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "global_step": self.global_step,
            "vla_actor_state_dict": self.vla_actor.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        print(f"  Checkpoint saved: {path}")

    def _get_dim_params(self) -> dict:
        sample = self.dataset[0]
        dim_params = {}
        if "action" in sample:
            dim_params["action_dim"] = sample["action"].shape[-1]
        for key in ("observation.state", "proprioception"):
            if key in sample:
                dim_params["proprioception_dim"] = sample[key].shape[-1]
                break
        return dim_params

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser(description="VLA Pretrain (Multi-GPU DDP)")
    parser.add_argument("--psi0", action="store_true", help="Use mock VLM + real Psi0 data")
    parser.add_argument("--mock", action="store_true", help="Use mock VLM + mock data")
    parser.add_argument("--data_root", type=str,
                        default="/vepfs/users/zza/hvla/data/psi-data-shared/unitree_dex3_converted/G1_Dex3_PickApple")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--head", type=str, default="regression", choices=["regression", "diffusion"])
    parser.add_argument("--action_dim", type=int, default=36)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--frames_dir", type=str, default="", help="Pre-extracted frames directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/vla_pretrain_ddp/")
    parser.add_argument("--log_dir", type=str, default="logs/vla_pretrain_ddp/")
    args = parser.parse_args()

    runner = VLAPretrainDDPRunner(args)
    runner.learn()


if __name__ == "__main__":
    main()
