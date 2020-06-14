from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


model_urls = {
    'peleenet_v1': './checkpoints/pretrained/lijun/peleenet.pth',
}


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features, growth_rate, bn_size, drop_rate,
    ):
        super().__init__()

        growth_rate = int(growth_rate / 2)
        inter_channel = int(growth_rate * bn_size / 4) * 4

        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4

        self.branch1a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
        self.branch1b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)

        self.branch2a = BasicConv2d(num_input_features, inter_channel, kernel_size=1)
        self.branch2b = BasicConv2d(inter_channel, growth_rate, kernel_size=3, padding=1)
        self.branch2c = BasicConv2d(growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        branch1 = self.branch1a(x)
        branch1 = self.branch1b(branch1)

        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)
        branch2 = self.branch2c(branch2)
        out = torch.cat([x, branch1, branch2], 1)

        return out


class _DenseBlock(nn.Sequential):
    _version = 2

    def __init__(
        self, num_layers, num_input_features, bn_size, growth_rate, drop_rate,
    ):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)


class _StemBlock(nn.Module):

    def __init__(self, num_input_channels, num_init_features):
        super().__init__()

        num_stem_features = int(num_init_features / 2)

        self.stem1 = BasicConv2d(
            num_input_channels, num_init_features, kernel_size=3,
            stride=2, padding=1,
        )
        self.stem2a = BasicConv2d(
            num_init_features, num_stem_features, kernel_size=1,
        )
        self.stem2b = BasicConv2d(
            num_stem_features, num_init_features, kernel_size=3,
            stride=2, padding=1,
        )
        self.stem3 = BasicConv2d(
            2 * num_init_features, num_init_features, kernel_size=1,
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2, ceil_mode=True,
        )

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = torch.cat([branch1, branch2], dim=1)
        out = self.stem3(out)

        return out


class PeleeNet(nn.Module):
    r"""PeleeNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>` and
    `"Pelee: A Real-Time Object Detection System on Mobile Devices" <https://arxiv.org/pdf/1804.06882.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (list of 4 ints) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(
        self,
        growth_rate=32,
        block_config=(3, 4, 8, 6),
        num_init_features=32,
        bn_size=(1, 2, 4, 4),
        drop_rate=0.05,
        num_classes=1000,
        memory_efficient=False,
    ):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ('stemblock', _StemBlock(3, num_init_features)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size[i],
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans = BasicConv2d(num_features, num_features, kernel_size=1, stride=1, padding=0)
            self.features.add_module('transition%d' % (i + 1), trans)

            if i != len(block_config) - 1:
                trans_pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
                self.features.add_module('transition%d_pool' % (i + 1), trans_pool)
                num_features = num_features

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.drop_rate = drop_rate

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.classifier(out)
        return out


def _peleenet(
    arch, growth_rate, block_config, num_init_features, bn_size,
    pretrained, progress, **kwargs,
):
    model = PeleeNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        state_dict = torch.load(model_urls[arch], map_location='cpu')
        model.features.load_state_dict(state_dict)
    return model


def peleenet_v1(pretrained=False, progress=True, **kwargs):
    r"""PeleeNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>` and
    `"Pelee: A Real-Time Object Detection System on Mobile Devices" <https://arxiv.org/pdf/1804.06882.pdf>`

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _peleenet(
        'peleenet_v1', 32, (3, 4, 8, 6), 32, (1, 2, 4, 4),
        pretrained, progress, **kwargs,
    )
