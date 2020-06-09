import torch
import torch.nn as nn

from torchvision.models.mobilenet import InvertedResidual, mobilenet_v2

from .backbone.backbone_utils import Backbone
from .box_heads.multibox_head import MultiBoxHeads


class SSDLiteWithMobileNetV2(nn.Module):
    r"""MobileNet V2 SSD model class
    Args:
        body: MobileNet V2 body layers
        extras: extra layers
        head: "multibox head" consists of loc and conf conv layers
    """
    def __init__(self, body, extras, head, **kwargs):
        super().__init__()

        self.features = body.features
        self.extras = nn.ModuleList(extras)
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

        for k, feature in enumerate(self.features):
            if k == 13:
                assert isinstance(feature, InvertedResidual)
                for j, sub_feature in enumerate(feature.conv):
                    samples = sub_feature(samples)
                    if j == 0:
                        sources.append(samples)
            else:
                samples = feature(samples)

        sources.append(samples)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            samples = v(samples)
            sources.append(samples)

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
    backbone = mobilenet_v2(pretrained=pretrained)
    return_layers = {'features': '0'}
    model = Backbone(backbone, return_layers)

    return model.body


def build_extras(in_channels):
    layers = list()

    channels = [in_channels, 512, 256, 256, 64]
    expand_ratio = [0.2, 0.25, 0.5, 0.25]

    for i in range(len(expand_ratio)):
        layers.append(InvertedResidual(channels[i], channels[i + 1], 2, expand_ratio[i]))

    return layers


class SeperableConv2d(nn.Sequential):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d."""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super().__init__(
            nn.Conv2d(in_planes, in_planes, kernel_size, stride=stride, padding=padding, groups=in_planes),
            norm_layer(in_planes),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_planes, out_planes, 1),
        )


def build_multibox(cfg, num_classes, width_mult=1.0):
    loc_layers = list()
    conf_layers = list()

    in_channels = round(576 * width_mult)
    channels = [in_channels, 1280, 512, 256, 256, 64]

    for i in range(5):
        loc_layers.append(SeperableConv2d(channels[i], cfg[i] * 4, 3, padding=1))
        conf_layers.append(SeperableConv2d(channels[i], cfg[i] * num_classes, 3, padding=1))

    loc_layers.append(nn.Conv2d(channels[-1], cfg[-1] * 4, 1))
    conf_layers.append(nn.Conv2d(channels[-1], cfg[-1] * num_classes, 1))

    return (loc_layers, conf_layers)


model_urls = {'ssd_lite_mobilenet_v2': ''}


def build(args):
    if args.image_size != 300:
        raise NotImplementedError(
            "You specified image_size [{}]. However, currently only "
            "MobileNetV2SSD300 (image_size=300) is supported!".format(args.image_size),
        )

    pretrained_backbone = False if args.pretrained else True
    body_layers = build_backbone(pretrained=pretrained_backbone)
    extras_layers = build_extras(1280)

    anchor_nms_cfg = [6, 6, 6, 6, 6, 6]  # number of boxes per feature map location
    head_layers = build_multibox(anchor_nms_cfg, args.num_classes)

    model = SSDLiteWithMobileNetV2(
        body_layers,
        extras_layers,
        head_layers,
        image_size=args.image_size,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        feature_maps=[19, 10, 5, 3, 2, 1],
        min_ratio=20,
        max_ratio=80,
        steps=[16, 32, 64, 100, 150, 300],
        clip=True,
        min_sizes=[60, 105, 150, 195, 240, 285],
        max_sizes=[105, 150, 195, 240, 285, 330],
    )

    return model
