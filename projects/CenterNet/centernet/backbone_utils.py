from collections import OrderedDict

import torch.nn as nn
import torchvision.models


class Backbone(nn.Module):
    def __init__(
        self, backbone_name, feature_name='layer4',
        pretrained=True,
    ):
        super().__init__()
        self.body = ConvFeaturesGetter(
            backbone_name,
            feature_name=feature_name,
            pretrained=pretrained,
        )

    def forward(self, x):
        return self.body(x)


class ConvFeaturesGetter(nn.Module):
    """CNN features getter for the Encoder of image."""
    def __init__(
        self, backbone_name, feature_name='layer4',
        pretrained=True,
    ):
        super().__init__()
        # loading network
        layers = OrderedDict()

        backbone = getattr(torchvision.models, backbone_name)(
            pretrained=pretrained,
        )

        for name, module in backbone.named_children():
            layers[name] = module
            if name == feature_name:
                break

        self.layers = nn.ModuleDict(layers)

    def forward(self, x):
        for _, module in self.layers.items():
            x = module(x)
        return x
