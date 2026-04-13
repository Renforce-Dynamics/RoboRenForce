from __future__ import annotations

import torch
from RoboRenForce import configclass

class LipsCfgMixin:
    r"""Mixin for Lipschitz-related configuration parameters.

    This mixin provides the hyper-parameter for the Lipschitz
    regularization term used to control the sensitivity of the policy
    with respect to its input state.

    Given a deterministic policy output :math:`\pi_\theta(s)` (e.g.
    the mean of a Gaussian policy), the Lipschitz loss is typically
    defined as
    $$
    L_{\text{Lips}} \;=\;
    \lambda_l \, \mathbb{E}_s \big\|
    \nabla_s \pi_\theta(s) \big\|_2,
    $$
    where :attr:`lips_lambda` is the regularization weight
    :math:`\lambda_l`. Penalizing the gradient norm enforces
    **Lipschitz continuity** so that similar inputs induce similar
    actions.
    """

    # Lipschitz specific parameter
    lips_lambda: float = 5e-3  # Lipschitz regularization weight


class LipsLossMixin:
    r"""Mixin providing Lipschitz loss computation interface.

    This mixin implements the Lipschitz loss described in
    :class:`LipsCfgMixin` and exposes a single method
    :meth:`compute_lips_loss`.

    For a batch of observations :math:`s` and deterministic policy
    output :math:`\pi_\theta(s)` (taken as the action mean), it
    approximates the gradient norm
    $$
    \big\| \nabla_s \pi_\theta(s) \big\|_2
    $$
    by computing Jacobian-vector products for each action dimension and
    aggregating the resulting norms across dimensions and batch.

    The returned scalar loss is
    $$
    L_{\text{Lips}} \;=\;
    \lambda_l \, \mathbb{E}_s \big\|
    \nabla_s \pi_\theta(s) \big\|_2.
    $$

    The subclass is expected to define:

    - :attr:`actor`: policy network with :meth:`act`, whose
      :attr:`action_mean` field holds :math:`\pi_\theta(s)`.
    - attribute :attr:`lips_lambda` (usually provided by
      :class:`LipsCfgMixin`).
    """

    lips_lambda: float

    def __init__(self, cfg: LipsCfgMixin, *args, **kwargs):
        # Initialize parent classes first
        super().__init__(cfg, *args, **kwargs)

        # Initialize Lipschitz parameter from cfg (if present)
        self.lips_lambda = float(getattr(cfg, "lips_lambda", 0.0))

    def compute_lips_loss(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Compute Lipschitz regularization loss.

        Args:
            obs_batch: [batch_size, obs_dim] policy observations.

        Returns:
            lips_loss: scalar tensor containing Lipschitz loss.
        """
        lips_loss = torch.zeros((), device=obs_batch.device, dtype=obs_batch.dtype)

        if self.lips_lambda <= 0:
            return lips_loss

        # Enable gradient computation for input observations
        obs_batch_grad = obs_batch.clone().detach().requires_grad_(True)

        # Forward pass through actor to get policy mean (do NOT sample actions here)
        _ = self.actor(obs_batch_grad)
        policy_mean = self.actor.action_mean

        # Compute gradients of policy output w.r.t. input states
        batch_size, action_dim = policy_mean.shape
        grad_norms = []

        for i in range(action_dim):
            grad_outputs = torch.zeros_like(policy_mean)
            grad_outputs[:, i] = 1.0

            grads = torch.autograd.grad(
                outputs=policy_mean,
                inputs=obs_batch_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            grad_norm = torch.norm(grads, dim=-1)  # [batch_size]
            grad_norms.append(grad_norm)

        grad_norms = torch.stack(grad_norms, dim=-1)  # [batch_size, action_dim]
        mean_grad_norm = torch.mean(grad_norms, dim=-1)  # [batch_size]

        lips_loss = self.lips_lambda * torch.mean(mean_grad_norm)

        return lips_loss

