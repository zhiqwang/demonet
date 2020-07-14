# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Zhiqiang Wang (zhiqwang@outlook.com)
"""
Backbone modules.
"""

import torch

from torch import nn

from torchvision.models.mobilenet import InvertedResidual, mobilenet_v2
from torchvision.models._utils import IntermediateLayerGetter

from typing import List

from util.misc import NestedTensor, is_main_process


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_layers: dict,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "features" not in name:
                parameter.requires_grad_(False)

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        samples = tensor_list.tensors
        features = []

        for k, feature in enumerate(self.body):
            if k == 14:
                for j, sub_feature in enumerate(feature.conv):
                    samples = sub_feature(samples)
                    if j == 0:
                        features.append(samples)
            else:
                samples = feature(samples)

        features.append(samples)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            samples = v(samples)
            features.append(samples)

        return features


class BackboneWithMobileNet(BackboneBase):
    """MobileNet backbone with frozen BatchNorm."""
    def __init__(
        self,
        train_backbone: bool,
    ):
        backbone = mobilenet_v2(pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        return_layers = {"features": "0"}
        num_channels = 1280
        super().__init__(backbone, return_layers, train_backbone, num_channels)


class ExtraLayers(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        extra_layers = []

        hidden_dims = [512, 256, 256, 64]
        expand_ratios = [0.2, 0.25, 0.5, 0.25]
        strides = [2, 2, 2, 2]

        for i in range(len(expand_ratios)):
            inp = hidden_dims[i - 1] if i > 0 else in_channels
            extra_layers.append(InvertedResidual(inp, hidden_dims[i], strides[i], expand_ratios[i]))

        self.extra_layers = nn.ModuleList(extra_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, extra_layers):
        super().__init__(backbone, extra_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        features = []
        for name, x in xs.items():
            out.append(x)
            # extra layers
            features.append(self[1](x).to(x.tensors.dtype))

        return out, features


def build_backbone(args):
    train_backbone = args.lr_backbone > 0
    backbone = BackboneWithMobileNet(train_backbone)
    extra_layers = ExtraLayers(backbone.num_channels)
    model = Joiner(backbone, extra_layers)

    return model
