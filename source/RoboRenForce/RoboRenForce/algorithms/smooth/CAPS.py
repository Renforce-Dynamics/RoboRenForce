from __future__ import annotations

import torch


class CAPSCfgMixin:
    r"""Mixin for CAPS-related configuration parameters.

    This mixin provides the hyper-parameters for the
    **Consistency-based Action Policy Smoothness (CAPS)** regularizer.

    CAPS penalizes fast changes of the deterministic policy mean
    $\mu_\theta(s)$ both **temporally** and **spatially**:

    - Temporal smoothness:
      $$
      L_T \;=\; \mathbb{E}_t \big\| \mu_\theta(s_t) - \mu_\theta(s_{t+1}) \big\|_2^2
      $$
    - Spatial smoothness (noisy perturbations in state space):
      $$
      L_S \;=\; \mathbb{E}_{s,\epsilon}
      \big\| \mu_\theta(s) - \mu_\theta(s + \sigma \epsilon) \big\|_2^2,
      \qquad \epsilon \sim \mathcal{N}(0, I)
      $$

    The total CAPS loss combined with PPO is typically
    $$
    L_{\text{CAPS}} \;=\; \lambda_T L_T \;+\; \lambda_S L_S,
    $$
    where :attr:`caps_lambda_t` and :attr:`caps_lambda_s` are the two
    weighting coefficients and :attr:`caps_sigma` controls the noise
    scale $\sigma$ used for spatial perturbations.
    """

    # CAPS specific parameters
    caps_lambda_t: float = 1e-2  # Temporal smoothness weight
    caps_lambda_s: float = 1e-2  # Spatial smoothness weight
    caps_sigma: float = 5e-2  # Noise scale for spatial perturbation


class CAPSLossMixin:
    r"""Mixin providing CAPS loss computation interface.

    This mixin implements the CAPS loss described in :class:`CAPSCfgMixin`
    and exposes a single method :meth:`compute_caps_loss` that can be
    added on top of any base RL loss.

    Given a batch of observations :math:`s` and the current deterministic
    policy mean :math:`\mu_\theta(s)`, the mixin computes

    - **spatial smoothness**
      $$
      L_S \;=\; \mathbb{E}_{s,\epsilon}
      \big\| \mu_\theta(s) - \mu_\theta(s + \sigma \epsilon) \big\|_2^2,
      \qquad \epsilon \sim \mathcal{N}(0, I)
      $$
    - **temporal smoothness** (approximated by a shifted mini-batch)
      $$
      L_T \;=\; \mathbb{E}_t
      \big\| \mu_\theta(s_t) - \mu_\theta(s_{t+1}) \big\|_2^2
      $$

    and returns the weighted sum
    $$
    L_{\text{CAPS}} \;=\; \lambda_T L_T \;+\; \lambda_S L_S.
    $$

    The subclass is expected to define:

    - :attr:`actor`: policy network with :meth:`act` and optionally
      :meth:`act_inference` returning deterministic actions / means.
    - :attr:`device`: torch device string.
    - attributes :attr:`caps_lambda_t`, :attr:`caps_lambda_s`,
      :attr:`caps_sigma` (usually provided by :class:`CAPSCfgMixin`).
    """

    caps_lambda_t: float
    caps_lambda_s: float
    caps_sigma: float
    device: str

    def __init__(self, cfg: CAPSCfgMixin, *args, **kwargs):
        # Initialize parent classes first
        super().__init__(cfg, *args, **kwargs)

        # Initialize CAPS parameters from cfg (if present)
        self.caps_lambda_t = float(getattr(cfg, "caps_lambda_t", 0.0))
        self.caps_lambda_s = float(getattr(cfg, "caps_lambda_s", 0.0))
        self.caps_sigma = float(getattr(cfg, "caps_sigma", 0.0))

    def compute_caps_loss(self, obs_batch: torch.Tensor, mu_batch: torch.Tensor) -> torch.Tensor:
        """Compute CAPS regularization loss.

        Args:
            obs_batch: [batch_size, obs_dim] observations used in the PPO update.
            mu_batch:  [batch_size, action_dim] current policy mean actions.

        Returns:
            caps_loss: scalar tensor containing total CAPS loss.
        """
        # Start with zero tensor on correct device for stable accumulation
        caps_loss = torch.zeros((), device=obs_batch.device, dtype=mu_batch.dtype)

        # Spatial smoothness: L_S = ||μ_batch - act_inference(obs_batch + σ*noise)||²
        if self.caps_lambda_s > 0:
            with_noise_obs = obs_batch + self.caps_sigma * torch.randn_like(obs_batch)

            # Get deterministic actions for perturbed observations
            if hasattr(self.actor, "act_inference"):
                actions_perturbed = self.actor.act_inference(with_noise_obs)
            else:
                # Fallback: use action_mean after forward pass
                self.actor.act(with_noise_obs)
                actions_perturbed = self.actor.action_mean

            caps_s_loss = torch.square(torch.norm(mu_batch - actions_perturbed, dim=-1)).mean()
            caps_loss = caps_loss + self.caps_lambda_s * caps_s_loss

        # Temporal smoothness: L_T = ||μ_batch - act_inference(next_obs_batch)||²
        # Note: we approximate next_obs_batch by shifting the current mini-batch.
        if self.caps_lambda_t > 0:
            batch_size = obs_batch.shape[0]
            if batch_size > 1:
                next_obs_approx = torch.cat([obs_batch[1:], obs_batch[-1:]], dim=0)

                if hasattr(self.actor, "act_inference"):
                    actions_next = self.actor.act_inference(next_obs_approx)
                else:
                    self.actor.act(next_obs_approx)
                    actions_next = self.actor.action_mean

                caps_t_loss = torch.square(torch.norm(mu_batch - actions_next, dim=-1)).mean()
                caps_loss = caps_loss + self.caps_lambda_t * caps_t_loss

        return caps_loss