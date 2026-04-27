"""PolicyAdapter — wraps a torch.nn.Module so the InferenceWorker can call it.

The InferenceWorker expects a `policy` object with this minimal API:

    actions: torch.Tensor = policy.predict(obs_dict: dict) -> (B, chunk, action_dim)

`PolicyAdapter` provides exactly that surface around any of:
  - `RoboRenForce.components.actor.vla_actor.VLAActor`
  - any `nn.Module` whose forward signature matches `(obs_dict) -> Tensor`

Two responsibilities the adapter takes on:
  1. Move tensors to the policy's device (workers serialize CPU tensors over
     shared memory; the model lives on GPU in the inference worker).
  2. Return CPU tensors so the producer can stuff them back into a
     SharedTensorRef without a device round-trip in the env worker.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
import torch.nn as nn


class PolicyAdapter:
    """Wrap an nn.Module + a `predict_fn` so the InferenceWorker can use it.

    `predict_fn(model, obs_dict_on_device) -> Tensor` is the only piece that
    knows the model's actual forward signature. Default uses
    `model.get_action(obs_dict)` if present, else `model(obs_dict)`.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        predict_fn: Optional[Callable[[nn.Module, dict], torch.Tensor]] = None,
        return_cpu: bool = True,
    ):
        self._model = model.to(device).eval()
        self._device = torch.device(device)
        self._predict_fn = predict_fn or self._default_predict
        self._return_cpu = return_cpu

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> nn.Module:
        return self._model

    @torch.no_grad()
    def predict(self, obs_dict: dict) -> torch.Tensor:
        moved = self._move_to_device(obs_dict)
        out = self._predict_fn(self._model, moved)
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"PolicyAdapter: predict_fn must return Tensor, got {type(out).__name__}"
            )
        if self._return_cpu:
            out = out.detach().cpu()
        return out

    # ---- helpers -----------------------------------------------------------

    def _move_to_device(self, obs_dict: dict) -> dict:
        out: dict[str, Any] = {}
        for k, v in obs_dict.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self._device, non_blocking=False)
            else:
                out[k] = v
        return out

    @staticmethod
    def _default_predict(model: nn.Module, obs_dict: dict) -> torch.Tensor:
        if hasattr(model, "get_action"):
            return model.get_action(obs_dict)
        return model(obs_dict)
