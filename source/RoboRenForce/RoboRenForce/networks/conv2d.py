from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from dataclasses import MISSING

from RoboRenForce import configclass
from RoboRenForce.utils.template.module_base import ModuleBase, ModuleBaseCfg
from RoboRenForce.networks.mlp import MLP, MLPCfg


def conv2d_output_shape(h: int, w: int, kernel_size: Union[int, Tuple[int, int]] = 1, 
                        stride: Union[int, Tuple[int, int]] = 1, 
                        padding: Union[int, Tuple[int, int]] = 0, 
                        dilation: int = 1) -> Tuple[int, int]:
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride,) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding,) * 2
    h = (h + (2 * ph) - (dilation * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (dilation * (kw - 1)) - 1) // sw + 1
    return h, w


class Conv2dModel(ModuleBase):
    """
    2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1. Requires number of input channels, but
    not input shape. Uses ``torch.nn.Conv2d``.
    """

    def __init__(
        self,
        cfg: Conv2dModelCfg,
        in_channels: int,
    ):
        """
        Args:
            cfg: Configuration for the Conv2d model.
            in_channels: Number of input channels.
        """
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        
        # Parse configuration
        channels = cfg.channels
        kernel_sizes = cfg.kernel_sizes
        strides = cfg.strides
        paddings = cfg.paddings if cfg.paddings is not None else [0] * len(channels)
        use_maxpool = cfg.use_maxpool
        
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        
        in_channels_list = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        
        # Build conv layers
        conv_layers = [
            nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k, stride=s, padding=p)
            for (ic, oc, k, s, p) in zip(in_channels_list, channels, kernel_sizes, strides, paddings)
        ]
        
        sequence = []
        normlayer = cfg.normlayer
        if isinstance(normlayer, str):
            normlayer = getattr(nn, normlayer)
        
        for conv_layer, oc, maxp_stride in zip(conv_layers, channels, maxp_strides):
            sequence.append(conv_layer)
            if normlayer is not None:
                sequence.append(normlayer(oc))
            # Get activation from config
            if cfg.activation is not None:
                if isinstance(cfg.activation, str):
                    activation_cls = getattr(nn, cfg.activation)
                    sequence.append(activation_cls())
                else:
                    sequence.append(cfg.activation)
            if maxp_stride > 1:
                sequence.append(nn.MaxPool2d(maxp_stride))
        
        self.conv = nn.Sequential(*sequence)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W].
        """
        return self.conv(input)

    def conv_out_size(self, h: int, w: int, c: Optional[int] = None) -> int:
        """
        Helper function to return the output size for a given input shape,
        without actually performing a forward pass through the model.
        """
        current_h, current_w = h, w
        current_c = c
        
        for child in self.conv.children():
            try:
                current_h, current_w = conv2d_output_shape(
                    current_h, current_w, 
                    child.kernel_size, 
                    child.stride, 
                    child.padding
                )
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                current_c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        
        return current_h * current_w * current_c

    def conv_out_resolution(self, h: int, w: int) -> Tuple[int, int]:
        """
        Helper function that return the resolution (H, W) for a given input resolution.
        """
        current_h, current_w = h, w
        
        for child in self.conv.children():
            try:
                current_h, current_w = conv2d_output_shape(
                    current_h, current_w,
                    child.kernel_size,
                    child.stride,
                    child.padding
                )
            except AttributeError:
                pass  # Not a conv or maxpool layer.
        
        return current_h, current_w


@configclass
class Conv2dModelCfg(ModuleBaseCfg):
    """Configuration for Conv2dModel."""
    class_type: type[nn.Module] = Conv2dModel
    
    channels: List[int] = MISSING
    """List of output channels for each conv layer."""
    
    kernel_sizes: List[Union[int, Tuple[int, int]]] = MISSING
    """List of kernel sizes for each conv layer."""
    
    strides: List[Union[int, Tuple[int, int]]] = MISSING
    """List of strides for each conv layer."""
    
    paddings: Optional[List[Union[int, Tuple[int, int]]]] = None
    """List of paddings for each conv layer. If None, defaults to [0, ...]."""
    
    activation: Optional[Union[str, nn.Module]] = "ReLU"
    """Activation function. Can be string (e.g., 'ReLU') or nn.Module."""
    
    use_maxpool: bool = False
    """If True: convs use stride 1, maxpool downsample. If False: convs use specified stride."""
    
    normlayer: Optional[Union[str, type[nn.Module]]] = None
    """Normalization layer. Can be string (e.g., 'BatchNorm2d') or type."""


class Conv2dHeadModel(ModuleBase):
    """
    Model component composed of a ``Conv2dModel`` component followed by
    a fully-connected ``MLP`` head. Requires full input image shape to
    instantiate the MLP head.
    """

    def __init__(
        self,
        cfg: Conv2dHeadModelCfg,
        image_shape: Tuple[int, int, int],  # (C, H, W)
    ):
        """
        Args:
            cfg: Configuration for the Conv2dHead model.
            image_shape: Input image shape as (C, H, W).
        """
        super().__init__()
        self.cfg = cfg
        c, h, w = image_shape
        
        # Build conv component
        conv_cfg = Conv2dModelCfg(
            channels=cfg.channels,
            kernel_sizes=cfg.kernel_sizes,
            strides=cfg.strides,
            paddings=cfg.paddings,
            activation=cfg.activation,
            use_maxpool=cfg.use_maxpool,
            normlayer=cfg.normlayer,
        )
        self.conv = conv_cfg.construct_from_cfg(in_channels=c)
        
        # Calculate conv output size
        conv_out_size = self.conv.conv_out_size(h, w)
        
        # Build MLP head if needed
        if cfg.head_cfg is not None:
            # Determine output size
            if cfg.output_size is not None:
                out_size = cfg.output_size
            elif cfg.head_cfg.hidden_features:
                out_size = cfg.head_cfg.hidden_features[-1]
            else:
                out_size = conv_out_size
            
            # Build head with proper hidden features
            hidden_features = cfg.head_cfg.hidden_features
            if cfg.output_size is not None and (not hidden_features or hidden_features[-1] != cfg.output_size):
                hidden_features = hidden_features + [cfg.output_size]
            
            # Adjust activations to match number of layers
            activations = cfg.head_cfg.activations
            num_layers = len(hidden_features) + 1  # +1 for input->first hidden
            if len(activations) < num_layers:
                activations = activations + [[]] * (num_layers - len(activations))
            elif len(activations) > num_layers:
                activations = activations[:num_layers]
            
            # Create modified head config
            head_cfg = cfg.head_cfg.replace(
                hidden_features=hidden_features,
                activations=activations,
            )
            
            self.head = head_cfg.construct_from_cfg(
                in_feature=conv_out_size,
                out_feature=out_size
            )
            self._output_size = out_size
        else:
            self.head = lambda x: x
            self._output_size = conv_out_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute the convolution and fully connected head on the input;
        assumes correct input shape: [B,C,H,W].
        """
        conv_out = self.conv(input)
        flattened = conv_out.view(input.shape[0], -1)
        return self.head(flattened)

    @property
    def output_size(self) -> int:
        """Returns the final output size after MLP head."""
        return self._output_size


@configclass
class Conv2dHeadModelCfg(ModuleBaseCfg):
    """Configuration for Conv2dHeadModel."""
    class_type: type[nn.Module] = Conv2dHeadModel
    
    channels: List[int] = MISSING
    """List of output channels for each conv layer."""
    
    kernel_sizes: List[Union[int, Tuple[int, int]]] = MISSING
    """List of kernel sizes for each conv layer."""
    
    strides: List[Union[int, Tuple[int, int]]] = MISSING
    """List of strides for each conv layer."""
    
    paddings: Optional[List[Union[int, Tuple[int, int]]]] = None
    """List of paddings for each conv layer."""
    
    activation: Optional[Union[str, nn.Module]] = "ReLU"
    """Activation function."""
    
    use_maxpool: bool = False
    """If True: convs use stride 1, maxpool downsample."""
    
    normlayer: Optional[Union[str, type[nn.Module]]] = None
    """Normalization layer."""
    
    head_cfg: Optional[MLPCfg] = None
    """Configuration for the MLP head. If None, no head is added."""
    
    output_size: Optional[int] = None
    """Output size of the head. If None and head_cfg is provided, uses last hidden size."""
