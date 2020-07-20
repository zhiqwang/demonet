# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Zhiqiang Wang (zhiqwang@outlook.com)
"""
Backbone modules.
"""

from torch import nn, Tensor

from torchvision.models.mobilenet import InvertedResidual, mobilenet_v2
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor

from torch.jit.annotations import List


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
            if not train_backbone:
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
        backbone = mobilenet_v2(pretrained=True).features
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
