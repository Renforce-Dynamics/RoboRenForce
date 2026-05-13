from __future__ import annotations

import torch
from torch import nn
from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBaseCfg

class NormalizerEmpirical(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, cfg: NormalizerEmpirical, shape: int):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = cfg.eps
        self.until = cfg.until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        # count is a buffer so load_state_dict restores it; otherwise update()
        # after load runs rate = count_x/(0+count_x) = 1.0 and overwrites the
        # loaded running stats with a single batch.
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Backward-compat: older checkpoints saved before count was a buffer
        # don't have prefix+"count" in their state_dict. Treat their loaded
        # stats as well-estimated (set count high so subsequent update() steps
        # barely perturb _mean/_var) rather than starting count at 0.
        count_key = prefix + "count"
        if count_key not in state_dict:
            state_dict[count_key] = torch.tensor(int(1e8), dtype=torch.long)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        # Floor the divisor at sqrt(eps) by adding eps inside the sqrt rather
        # than to std directly. With default eps=1e-2 this caps amplification
        # on a collapsed dim (var=0) at 1/sqrt(eps)=10x instead of the
        # 1/eps=100x of the original (x - mean) / (std + eps) formula, which
        # could escalate to inf/nan through a few Linear+activation layers.
        return (x - self._mean) / torch.sqrt(self._var + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count.item() >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count.item()

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._var.clamp_(min=0.0)
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        # Mirror the floored divisor used in forward().
        return y * torch.sqrt(self._var + self.eps) + self._mean

@configclass
class NormalizerEmpiricalCfg(ModuleBaseCfg):
    class_type  : type[NormalizerEmpirical] = NormalizerEmpirical
    eps         : float = 1e-2 
    until       : int = None