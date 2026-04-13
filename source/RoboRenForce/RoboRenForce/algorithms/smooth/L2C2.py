from __future__ import annotations

import torch
from RoboRenForce import configclass

class L2C2CfgMixin:
    r"""Mixin for L2C2-related configuration parameters.

    This mixin provides the hyper-parameters for the
    **L2C2 (Lipschitz-Constrained Policy and Critic)** regularizer.

    L2C2 encourages both the policy mean :math:`\mu_\theta` and the
    value function :math:`V_\phi` to change smoothly along interpolations
    between consecutive states.

    For two consecutive observations :math:`s` and :math:`s'`, we form
    an interpolated state
    $$
    \tilde{s} \;=\; s + \alpha (s' - s), \qquad
    \alpha \in [-1, 1],
    $$
    and define:

    - **policy smoothness**
      $$
      L_{s,\pi} \;=\; \mathbb{E}
      \big\| \mu_\theta(s) - \mu_\theta(\tilde{s}) \big\|_2^2
      $$
    - **value smoothness**
      $$
      L_{s,V} \;=\; \mathbb{E}
      \big\| V_\phi(s) - V_\phi(\tilde{s}) \big\|_2^2
      $$

    The combined L2C2 loss is
    $$
    L_{\text{L2C2}} \;=\;
    \lambda_\pi L_{s,\pi} \;+\; \lambda_V L_{s,V},
    $$
    where :attr:`l2c2_lambda_pi` and :attr:`l2c2_lambda_v` weight policy
    and value smoothness respectively.
    """

    # L2C2 specific parameters
    l2c2_lambda_pi: float = 5e-3  # Policy smoothness weight
    l2c2_lambda_v: float = 2.5e-3  # Value function smoothness weight


class L2C2LossMixin:
    r"""Mixin providing L2C2 loss computation interface.

    This mixin implements the L2C2 loss described in :class:`L2C2CfgMixin`
    and exposes a single method :meth:`compute_l2c2_loss` that can be
    added on top of a base RL loss.

    Given a batch of policy observations :math:`s`, critic observations
    :math:`s_c`, current policy means :math:`\mu_\theta(s)` and value
    predictions :math:`V_\phi(s_c)`, the mixin:

    1. Approximates the **next** observations by shifting the mini-batch:
       $s' \approx \text{shift}(s)$, $s_c' \approx \text{shift}(s_c)$.
    2. Samples interpolation weights :math:`\alpha \in [-1, 1]` and builds
       $$
       \tilde{s} = s + \alpha (s' - s), \qquad
       \tilde{s}_c = s_c + \alpha (s_c' - s_c).
       $$
    3. Computes
       $$
       L_{s,\pi} = \mathbb{E}
       \big\| \mu_\theta(s) - \mu_\theta(\tilde{s}) \big\|_2^2,
       \qquad
       L_{s,V} = \mathbb{E}
       \big\| V_\phi(s_c) - V_\phi(\tilde{s}_c) \big\|_2^2.
       $$

    The returned loss is
    $$
    L_{\text{L2C2}} = \lambda_\pi L_{s,\pi} + \lambda_V L_{s,V}.
    $$

    The subclass is expected to define:

    - :attr:`actor`: policy network with :meth:`act` and optionally
      :meth:`act_inference`.
    - :attr:`critic`: value network taking critic observations.
    - :attr:`device`: torch device string.
    - attributes :attr:`l2c2_lambda_pi`, :attr:`l2c2_lambda_v` (usually
      provided by :class:`L2C2CfgMixin`).
    """

    l2c2_lambda_pi: float
    l2c2_lambda_v: float
    device: str

    def __init__(self, cfg: L2C2CfgMixin, *args, **kwargs):
        # Initialize parent classes first
        super().__init__(cfg, *args, **kwargs)

        # Initialize L2C2 parameters from cfg (if present)
        self.l2c2_lambda_pi = float(getattr(cfg, "l2c2_lambda_pi", 0.0))
        self.l2c2_lambda_v = float(getattr(cfg, "l2c2_lambda_v", 0.0))

    def compute_l2c2_loss(
        self,
        obs_batch: torch.Tensor,
        critic_obs_batch: torch.Tensor,
        mu_batch: torch.Tensor,
        value_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute L2C2 regularization loss.

        Args:
            obs_batch:        [batch_size, obs_dim] policy observations.
            critic_obs_batch: [batch_size, critic_obs_dim] critic observations.
            mu_batch:         [batch_size, action_dim] current policy mean actions.
            value_pred:       [batch_size, 1] current value predictions.

        Returns:
            l2c2_loss: scalar tensor containing total L2C2 loss.
        """
        l2c2_loss = torch.zeros((), device=obs_batch.device, dtype=value_pred.dtype)

        if self.l2c2_lambda_pi <= 0 and self.l2c2_lambda_v <= 0:
            return l2c2_loss

        batch_size = obs_batch.shape[0]
        if batch_size <= 1:
            return l2c2_loss

        # Approximate next observations by shifting in the batch
        next_obs_batch = torch.cat([obs_batch[1:], obs_batch[-1:]], dim=0)
        next_critic_obs_batch = torch.cat([critic_obs_batch[1:], critic_obs_batch[-1:]], dim=0)

        # Continuation mask (here assume all transitions are continuing)
        cont_batch = torch.ones(batch_size, 1, device=obs_batch.device, dtype=obs_batch.dtype)

        # Interpolation weights: mix_weights = cont_batch * (rand - 0.5) * 2.0
        mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0

        # Interpolated observations
        mix_obs_batch = obs_batch + mix_weights * (next_obs_batch - obs_batch)
        mix_critic_obs_batch = critic_obs_batch + mix_weights * (next_critic_obs_batch - critic_obs_batch)

        # Policy smoothness term
        if self.l2c2_lambda_pi > 0:
            if hasattr(self.actor, "act_inference"):
                actions_interpolated = self.actor.act_inference(mix_obs_batch)
            else:
                self.actor.act(mix_obs_batch)
                actions_interpolated = self.actor.action_mean

            policy_smooth_loss = torch.square(torch.norm(mu_batch - actions_interpolated, dim=-1)).mean()
            l2c2_loss = l2c2_loss + self.l2c2_lambda_pi * policy_smooth_loss

        # Value smoothness term
        if self.l2c2_lambda_v > 0:
            values_interpolated = self.critic(mix_critic_obs_batch)
            value_smooth_loss = torch.square(torch.norm(value_pred - values_interpolated, dim=-1)).mean()
            l2c2_loss = l2c2_loss + self.l2c2_lambda_v * value_smooth_loss

        return l2c2_loss