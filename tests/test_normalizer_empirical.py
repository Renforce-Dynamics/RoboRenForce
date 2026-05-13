"""Tests for the NormalizerEmpirical fix (count buffer + collapsed-std safety).

Covers the two regressions documented at
`source/RoboRenForce/RoboRenForce/components/normalizer/normalizer_empirical.py`:

1. `count` was a plain int, not a registered buffer. After `load_state_dict`
   the count reset to 0, so the first training-mode forward had
   `rate = batch / (0 + batch) = 1.0` and overwrote the loaded running stats
   with a single batch.

2. `forward` divided by `(_std + eps)` with `eps=1e-2`. Dims whose running
   variance had collapsed to 0 (e.g. obs that never change under zero
   randomization) got amplified by `1/eps = 100x`, escalating to inf/nan
   through a few Linear+activation layers.
"""

from __future__ import annotations

import torch

from RoboRenForce.components.normalizer.normalizer_empirical import (
    NormalizerEmpirical,
    NormalizerEmpiricalCfg,
)


def _make(shape: int = 4, eps: float = 1e-2) -> NormalizerEmpirical:
    cfg = NormalizerEmpiricalCfg(eps=eps)
    return NormalizerEmpirical(cfg, shape)


def test_count_is_a_buffer():
    n = _make()
    assert "count" in dict(n.named_buffers()), "count must be a registered buffer"
    assert n.count.dtype == torch.long


def test_count_survives_state_dict_round_trip():
    n1 = _make(shape=3)
    n1.train()
    for _ in range(5):
        n1(torch.randn(128, 3))
    saved_count = int(n1.count.item())
    assert saved_count == 5 * 128

    n2 = _make(shape=3)
    n2.load_state_dict(n1.state_dict())
    assert int(n2.count.item()) == saved_count, (
        f"count was not restored from state_dict: got {int(n2.count.item())}, "
        f"expected {saved_count}"
    )


def test_loaded_stats_are_not_overwritten_by_first_train_forward():
    """Regression for layer-1 bug: post-load update() must NOT use rate=1.0."""
    n1 = _make(shape=3)
    n1.train()
    for _ in range(20):
        n1(torch.randn(256, 3) * 2.0 + 1.0)
    saved_mean = n1._mean.clone()
    saved_var = n1._var.clone()

    n2 = _make(shape=3)
    n2.load_state_dict(n1.state_dict())
    n2.train()
    # A single very-different batch should barely move the loaded stats
    # because count is large.
    n2(torch.zeros(8, 3))
    drift_mean = (n2._mean - saved_mean).abs().max().item()
    drift_var = (n2._var - saved_var).abs().max().item()
    assert drift_mean < 0.05, f"loaded mean was overwritten (drift={drift_mean:.3f})"
    assert drift_var < 0.05, f"loaded var was overwritten (drift={drift_var:.3f})"


def test_collapsed_std_does_not_explode():
    """Regression for layer-2 bug: var=0 dim must not give 100x amplification."""
    n = _make(shape=2, eps=1e-2)
    # Force dim 0 into the collapsed state (constant input → var=0).
    n.train()
    constant_x = torch.zeros(64, 2)
    constant_x[:, 1] = 0.5  # dim 1 also constant, but at 0.5
    for _ in range(50):
        n(constant_x)
    assert n._var[0, 0].item() < 1e-6, "dim 0 should have collapsed"
    assert n._var[0, 1].item() < 1e-6, "dim 1 should have collapsed"

    n.eval()
    probe = torch.tensor([[1.0, 1.5]])
    out = n(probe)
    # With the fix: divisor = sqrt(0 + eps) = 0.1, so amplification capped at 10x.
    # With the old bug: divisor = 0 + eps = 0.01, amplification 100x.
    assert out.abs().max().item() < 15.0, (
        f"collapsed-dim amplification too large: out={out.tolist()} (expect ~10x at most)"
    )
    assert not torch.isnan(out).any(), "collapsed-dim forward produced NaN"
    assert not torch.isinf(out).any(), "collapsed-dim forward produced inf"


def test_old_checkpoint_without_count_buffer_loads():
    """Backward-compat: state_dict lacking `count` key must still load and behave sanely."""
    n1 = _make(shape=3)
    n1.train()
    for _ in range(10):
        n1(torch.randn(128, 3) * 2.0 + 1.0)
    saved_mean = n1._mean.clone()
    saved_var = n1._var.clone()

    sd_old = {k: v for k, v in n1.state_dict().items() if not k.endswith("count")}
    assert not any(k.endswith("count") for k in sd_old)

    n2 = _make(shape=3)
    # Should not raise (count defaulted to large value internally).
    n2.load_state_dict(sd_old)
    assert int(n2.count.item()) >= int(1e7), (
        "missing-count fallback should default count to a large value"
    )

    # Same drift check as test_loaded_stats_are_not_overwritten...
    n2.train()
    n2(torch.zeros(8, 3))
    drift_mean = (n2._mean - saved_mean).abs().max().item()
    drift_var = (n2._var - saved_var).abs().max().item()
    assert drift_mean < 0.05 and drift_var < 0.05


def test_normal_dim_normalization_is_unchanged():
    """Sanity: non-collapsed dims should normalize approximately to N(0, 1)."""
    n = _make(shape=2)
    n.train()
    for _ in range(100):
        n(torch.randn(512, 2) * 3.0 - 1.0)
    n.eval()
    x = torch.randn(4096, 2) * 3.0 - 1.0
    out = n(x)
    assert abs(out.mean().item()) < 0.1
    assert 0.85 < out.std().item() < 1.15
