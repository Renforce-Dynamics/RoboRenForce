"""
Tests for FusionLayer

IO tested:
    forward(vl_features, proprioception) -> fused_features
        Input:  vl_features (B, vl_dim), proprioception (B, proprio_dim)
        Output: fused (B, output_dim)

    Fusion types: "concat_mlp", "add"
"""

import torch
import pytest

from RoboRenForce.networks.vlm.fusion_layers import FusionLayer, FusionLayerCfg


B = 4
VL_DIM = 128
PROPRIO_DIM = 48
OUTPUT_DIM = 64


@pytest.fixture
def dim_params():
    return {"vl_feature_dim": VL_DIM, "proprio_dim": PROPRIO_DIM}


@pytest.fixture
def inputs():
    return (
        torch.randn(B, VL_DIM),
        torch.randn(B, PROPRIO_DIM),
    )


class TestConcatMLPFusion:
    def test_output_shape(self, dim_params, inputs):
        cfg = FusionLayerCfg(
            fusion_type="concat_mlp",
            output_dim=OUTPUT_DIM,
            hidden_dims=[128],
        )
        layer = FusionLayer(cfg, dim_params)
        vl, proprio = inputs
        out = layer(vl, proprio)
        assert out.shape == (B, OUTPUT_DIM)

    def test_gradient_flow(self, dim_params, inputs):
        cfg = FusionLayerCfg(output_dim=OUTPUT_DIM, hidden_dims=[64])
        layer = FusionLayer(cfg, dim_params)
        vl, proprio = inputs
        vl.requires_grad_(True)
        out = layer(vl, proprio)
        out.sum().backward()
        assert vl.grad is not None

    def test_different_hidden_dims(self, dim_params, inputs):
        cfg = FusionLayerCfg(output_dim=32, hidden_dims=[256, 128])
        layer = FusionLayer(cfg, dim_params)
        vl, proprio = inputs
        out = layer(vl, proprio)
        assert out.shape == (B, 32)


class TestAddFusion:
    def test_output_shape(self, dim_params, inputs):
        cfg = FusionLayerCfg(fusion_type="add", output_dim=OUTPUT_DIM)
        layer = FusionLayer(cfg, dim_params)
        vl, proprio = inputs
        out = layer(vl, proprio)
        assert out.shape == (B, OUTPUT_DIM)


class TestFusionErrors:
    def test_unknown_type(self, dim_params):
        cfg = FusionLayerCfg(fusion_type="unknown")
        with pytest.raises(ValueError, match="Unknown fusion_type"):
            FusionLayer(cfg, dim_params)
