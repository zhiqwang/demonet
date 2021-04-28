# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
"""
Backbone modules.
"""

from torch import nn, Tensor

from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models._utils import IntermediateLayerGetter

from ..util.misc import NestedTensor

from typing import Callable, Optional, List


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


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


def build_backbone(args):
    train_backbone = args.lr_backbone > 0
    model = MobileNetWithExtraBlocks(train_backbone)

    return model
