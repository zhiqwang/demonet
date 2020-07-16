# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Zhiqiang Wang (zhiqwang@outlook.com)
"""
Backbone modules.
"""

import torch
from torch import nn, Tensor

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
        extra_blocks: nn.Module,
        train_backbone: bool,
        return_layers_backbone: dict,
        return_layers_extra_blocks: dict,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "features" not in name:
                parameter.requires_grad_(False)

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers_backbone)
        self.extra_blocks = IntermediateLayerGetter(extra_blocks, return_layers=return_layers_extra_blocks)

    def forward(self, tensor_list: NestedTensor):
        xs_body = self.body(tensor_list.tensors)
        out: List[Tensor] = []

        for name, x in xs_body.items():
            out.append(x)

        xs_extra_blocks = self.extra_blocks(out[-1])
        for name, x in xs_extra_blocks.items():
            out.append(x)
        return out


class MobileNetWithExtraBlocks(BackboneBase):
    """MobileNet backbone with extra blocks."""
    def __init__(
        self,
        train_backbone: bool,
    ):
        backbone = mobilenet_v2(pretrained=is_main_process(),
                                norm_layer=FrozenBatchNorm2d).features
        return_layers_backbone = {"13": "0", "18": "1"}

        num_channels = 1280
        hidden_dims = [512, 256, 256, 64]
        expand_ratios = [0.2, 0.25, 0.5, 0.25]
        strides = [2, 2, 2, 2]
        extra_blocks = ExtraBlocks(num_channels, hidden_dims, expand_ratios, strides)
        return_layers_extra_blocks = {"0": "2", "1": "3", "2": "4", "3": "5"}

        super().__init__(
            backbone,
            extra_blocks,
            train_backbone,
            return_layers_backbone,
            return_layers_extra_blocks,
        )


class ExtraBlocks(nn.Sequential):
    def __init__(self, in_channels, hidden_dims, expand_ratios, strides):
        extra_blocks = []

        for i in range(len(expand_ratios)):
            input_dim = hidden_dims[i - 1] if i > 0 else in_channels
            extra_blocks.append(InvertedResidual(input_dim, hidden_dims[i], strides[i], expand_ratios[i]))

        super().__init__(*extra_blocks)


def build_backbone(args):
    train_backbone = args.lr_backbone > 0
    model = MobileNetWithExtraBlocks(train_backbone)

    return model
