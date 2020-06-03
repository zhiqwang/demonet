import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone.backbone_utils import Backbone
from .backbone.peleenet import BasicConv2d, peleenet_v1
from .box_heads.multibox_head import MultiBoxHeads


class Pelee(nn.Module):
    r"""Pelee model class, based on
    `"Pelee: A Real-Time Object Detection System on Mobile Devices" <https://arxiv.org/pdf/1804.06882.pdf>`
    Args:
        body: PeleeNet body layers
        extras: extra layers
        resblock: res block layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """
    def __init__(self, body, extras, resblock, head, **kwargs):
        super().__init__()

        self.features = body.features

        self.extras = nn.ModuleList(extras)
        self.resblock = nn.ModuleList(resblock)

        self.loc_conv = nn.ModuleList(head[0])
        self.conf_conv = nn.ModuleList(head[1])

        self.multibox_heads = MultiBoxHeads(**kwargs)

    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

    def forward(self, samples, targets=None):
        """
        Arguments:
            samples (Tensor): samples to be processed
            targets (Tensor): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        sources = list()
        loc = list()
        conf = list()

        for k, feat in enumerate(self.features):
            samples = feat(samples)
            if k == 8:
                sources.append(samples)

        sources.append(samples)

        for k, v in enumerate(self.extras):
            samples = v(samples)
            if k % 2 == 1:
                sources.append(samples)

        for k, x in enumerate(sources):
            sources[k] = self.resblock[k](x)

        for (x, l, c) in zip(sources, self.loc_conv, self.conf_conv):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        detections, detector_losses = self.multibox_heads(loc, conf, targets)

        return self.eager_outputs(detector_losses, detections)


def build_backbone(pretrained=True, trainable_layers=1):
    """
    Constructs a specified PeleeNet backbone. Freezes the specified number of layers in the backbone.

    Arguments:
        pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_layers (int): number of trainable (not frozen) peleenet layers starting from final block.
    """
    backbone = peleenet_v1(pretrained=pretrained)

    # select layers that wont be frozen
    assert trainable_layers == 1
    layers_to_train = ['features']
    # freeze layers only if pretrained backbone is used
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'features': '0'}

    model = Backbone(backbone, return_layers)

    return model.body


class ConvReLU(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x):
        out = F.relu(self.conv(x), inplace=True)
        return out


def build_extras(i, batch_norm=False):
    layers = list()
    in_channels = i
    channels = [128, 256, 128, 256, 128, 256]
    stride = [1, 2, 1, 1, 1, 1]
    padding = [0, 1, 0, 0, 0, 0]

    for k, v in enumerate(channels):
        if k % 2 == 0:
            if batch_norm:
                layers.append(BasicConv2d(
                    in_channels, v, kernel_size=1, padding=padding[k],
                ))
            else:
                layers.append(ConvReLU(
                    in_channels, v, kernel_size=1, padding=padding[k],
                ))
        else:
            if batch_norm:
                layers.append(BasicConv2d(
                    in_channels, v, kernel_size=3, padding=padding[k],
                    stride=stride[k],
                ))
            else:
                layers.append(ConvReLU(
                    in_channels, v, kernel_size=3, padding=padding[k],
                    stride=stride[k],
                ))
        in_channels = v

    return layers


class ResBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.res1a = ConvReLU(in_channels, 128, kernel_size=1)
        self.res1b = ConvReLU(128, 128, kernel_size=3, padding=1)
        self.res1c = ConvReLU(128, 256, kernel_size=1)

        self.res2a = ConvReLU(in_channels, 256, kernel_size=1)

    def forward(self, x):
        out1 = self.res1a(x)
        out1 = self.res1b(out1)
        out1 = self.res1c(out1)

        out2 = self.res2a(x)
        out = out1 + out2
        return out


def build_resblock(nchannels):
    resblock_layers = list()
    for k, v in enumerate(nchannels):
        resblock_layers.append(ResBlock(v))
    return resblock_layers


def build_multibox(cfg, num_classes):
    loc_layers = list()
    conf_layers = list()

    for k, v in enumerate([256] * 5):
        loc_layers.append(nn.Conv2d(v, cfg[k] * 4, kernel_size=1))
        conf_layers.append(nn.Conv2d(v, cfg[k] * num_classes, kernel_size=1))

    return (loc_layers, conf_layers)


model_urls = {'pelee': ''}


def build(args):
    if args.image_size != 304:
        raise NotImplementedError(
            "You specified image_size [{}]. However, currently only "
            "Pelee304 (image_size=304) is supported!".format(args.image_size),
        )

    pretrained_backbone = False
    body_layers = build_backbone(pretrained_backbone)
    extras_layers = build_extras(704, batch_norm=True)
    nchannels = [512, 704, 256, 256, 256]
    anchor_nms_cfg = [6, 6, 6, 6, 6]  # number of boxes per feature map location
    resblock_layers = build_resblock(nchannels)
    head_layers = build_multibox(anchor_nms_cfg, args.num_classes)

    model = Pelee(
        body_layers,
        extras_layers,
        resblock_layers,
        head_layers,
        image_size=args.image_size,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        feature_maps=[19, 10, 5, 3, 1],
        min_ratio=15,
        max_ratio=90,
        steps=[16, 30, 60, 101, 304],
        clip=True,
    )

    return model
