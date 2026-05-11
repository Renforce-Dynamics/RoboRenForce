"""Resume contract tests for OnPolicyRunner.save / .load.

Direct unit test of the save/load dict contract without constructing a
full PPO runner (which would require a heavy vectorized env). We build
small stub objects that look like the bits :meth:`save` and :meth:`load`
read on ``self``, then invoke the methods unbound.

Tracked fields (must round-trip):
- actor_critic.state_dict
- obs_normalizer.state_dict
- critic_normalizer.state_dict      (new in 2026-05-11 fix)
- alg.learning_rate                 (new in 2026-05-11 fix; KL-adaptive)
- current_learning_iteration        (key existed but load() ignored it)

Tracked field with opt-in:
- optimizer.state_dict — only restored when load_optimizer=True
"""

from __future__ import annotations

import torch
import torch.nn as nn

from RoboRenForce.runners.on_policy.on_policy_runner import OnPolicyRunner


class _StubLogger:
  """save() defers the file write to ``logger.save_model``; minimal stub."""

  def save_model(self, saved_dict, path, _iter):
    torch.save(saved_dict, path)


class _StubAlg:
  def __init__(self, lr: float):
    self.learning_rate = lr
    self.optimizer = torch.optim.Adam([nn.Parameter(torch.zeros(3))], lr=lr)


class _RunnerStub:
  """Duck-typed stub. save/load only touch attribute access, so we don't
  need to satisfy ``nn.Module.__init__`` here — using OnPolicyRunner
  directly would force a full env/cfg construction."""

  def __init__(self, lr: float, iter_count: int):
    self.actor_critic = nn.Linear(4, 2)
    self.obs_normalizer = nn.Linear(4, 4)
    self.critic_normalizer = nn.Linear(8, 8)
    self.alg = _StubAlg(lr=lr)
    self.current_learning_iteration = iter_count
    self.logger = _StubLogger()


def _fresh_runner(lr: float, iter_count: int) -> "_RunnerStub":
  return _RunnerStub(lr=lr, iter_count=iter_count)


def _bump_module(mod: nn.Module, scale: float) -> None:
  with torch.no_grad():
    for p in mod.parameters():
      p.add_(scale)


def test_save_dict_has_all_tracked_keys(tmp_path):
  src = _fresh_runner(lr=2.5e-4, iter_count=1234)
  ckpt = tmp_path / "model.pt"

  OnPolicyRunner.save(src, str(ckpt), infos={"note": "test"})

  loaded = torch.load(ckpt)
  for key in (
    "model_state_dict",
    "obs_norm_state_dict",
    "critic_norm_state_dict",
    "optimizer_state_dict",
    "alg_learning_rate",
    "iter",
    "infos",
  ):
    assert key in loaded, f"save() dropped key {key!r}"

  assert loaded["alg_learning_rate"] == 2.5e-4
  assert loaded["iter"] == 1234
  assert loaded["infos"] == {"note": "test"}


def test_load_restores_critic_normalizer_and_iter(tmp_path):
  src = _fresh_runner(lr=1e-3, iter_count=500)
  _bump_module(src.actor_critic, 0.1)
  _bump_module(src.obs_normalizer, 0.2)
  _bump_module(src.critic_normalizer, 0.3)

  ckpt = tmp_path / "model.pt"
  OnPolicyRunner.save(src, str(ckpt))

  dst = _fresh_runner(lr=9.9e-9, iter_count=0)  # different starting state
  OnPolicyRunner.load(dst, str(ckpt))

  for (name, a), (_, b) in zip(
    src.actor_critic.state_dict().items(),
    dst.actor_critic.state_dict().items(),
  ):
    assert torch.allclose(a, b), f"actor_critic.{name} did not round-trip"

  for ((_, a), (_, b)) in zip(
    src.obs_normalizer.state_dict().items(),
    dst.obs_normalizer.state_dict().items(),
  ):
    assert torch.allclose(a, b), "obs_normalizer did not round-trip"

  for ((_, a), (_, b)) in zip(
    src.critic_normalizer.state_dict().items(),
    dst.critic_normalizer.state_dict().items(),
  ):
    assert torch.allclose(a, b), "critic_normalizer did not round-trip"

  assert dst.current_learning_iteration == 500, "iter not restored"
  assert dst.alg.learning_rate == 1e-3, "alg.learning_rate not restored"
  # optimizer's param_group lr should track the restored alg lr.
  assert dst.alg.optimizer.param_groups[0]["lr"] == 1e-3


def test_load_optimizer_default_is_off(tmp_path):
  """load_optimizer defaults to False so cross-task FT does not inherit
  Adam momentum from the source ckpt."""
  src = _fresh_runner(lr=1e-3, iter_count=10)
  # Step the optimizer once so it has non-trivial momentum state.
  src.alg.optimizer.param_groups[0]["params"][0].grad = torch.ones(3)
  src.alg.optimizer.step()
  src_opt_state = dict(src.alg.optimizer.state_dict())

  ckpt = tmp_path / "model.pt"
  OnPolicyRunner.save(src, str(ckpt))

  dst = _fresh_runner(lr=9e-9, iter_count=0)
  OnPolicyRunner.load(dst, str(ckpt))  # default load_optimizer=False

  assert dst._last_resume_status["optimizer"] == "skipped"
  # The optimizer's internal state buffer should still be empty (no step taken).
  assert dst.alg.optimizer.state_dict()["state"] == {}, (
    "optimizer state must not be inherited when load_optimizer=False"
  )

  # With explicit opt-in, optimizer is restored.
  dst2 = _fresh_runner(lr=9e-9, iter_count=0)
  OnPolicyRunner.load(dst2, str(ckpt), load_optimizer=True)
  assert dst2._last_resume_status["optimizer"] == "loaded"


def test_load_warns_on_missing_keys_for_old_ckpts(tmp_path):
  """Old ckpts (pre-fix) lack critic_norm and alg_learning_rate. Loading
  them must succeed and report ``missing`` in the status dict."""
  src = _fresh_runner(lr=5e-4, iter_count=100)
  ckpt = tmp_path / "old.pt"

  # Hand-write the legacy save shape: no critic_norm / alg_learning_rate.
  torch.save(
    {
      "model_state_dict": src.actor_critic.state_dict(),
      "obs_norm_state_dict": src.obs_normalizer.state_dict(),
      "optimizer_state_dict": src.alg.optimizer.state_dict(),
      "iter": src.current_learning_iteration,
      "infos": None,
    },
    ckpt,
  )

  dst = _fresh_runner(lr=9e-9, iter_count=0)
  OnPolicyRunner.load(dst, str(ckpt))
  status = dst._last_resume_status
  assert status["model"] == "loaded"
  assert status["obs_normalizer"] == "loaded"
  assert status["critic_normalizer"] == "missing"
  assert status["alg_learning_rate"] == "missing"
  assert status["iter"] == "loaded"
