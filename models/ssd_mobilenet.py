import warnings

import torch
from torch import nn, Tensor

from torchvision.models.mobilenet import InvertedResidual, mobilenet_v2

from util.misc import NestedTensor, is_main_process

from .backbone import BackboneBase
from .multibox_head import MultiBoxHeads

from torch.jit.annotations import List, Optional, Dict, Tuple


class SSDLiteWithMobileNetV2(nn.Module):
    r"""MobileNet V2 SSD model class
    Args:
        backbone: MobileNet V2 backbone layers
        extras: extra layers
        head: "multibox head" consists of box_regression and class_logits conv layers
    """
    def __init__(self, backbone, extras, head, onnx_export=False, **kwargs):
        super().__init__()

        self.backbone = backbone.body.features
        self.extras = nn.ModuleList(extras)
        self.bbox_pred = nn.ModuleList(head[0])
        self.cls_logits = nn.ModuleList(head[1])

        self.onnx_export = onnx_export

        self.multibox_heads = MultiBoxHeads(**kwargs)
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(
        self,
        losses,  # type: Dict[str, Tensor]
        detections,  # type: List[Dict[str, Tensor]]
    ):
        # type: (...) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(
        self,
        tensor_list,  # type: NestedTensor
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            tensor_list (NestedTensor): samples to be processed
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        samples = tensor_list.tensors

        features = list()
        box_regression = list()
        class_logits = list()

        for k, feature in enumerate(self.backbone):
            if k == 14:
                assert isinstance(feature, InvertedResidual)
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

        for (x, b, c) in zip(features, self.bbox_pred, self.cls_logits):
            box_regression.append(b(x).permute(0, 2, 3, 1).contiguous())
            class_logits.append(c(x).permute(0, 2, 3, 1).contiguous())

        detections, detector_losses = self.multibox_heads(
            box_regression, class_logits, features, targets=targets)

        return self.eager_outputs(detector_losses, detections)
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("DEMONET always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (detector_losses, detections)
        else:
            return self.eager_outputs(detector_losses, detections)


def build_backbone(train_backbone=True):
    """
    Constructs a specified MobileNet backbone. Freezes the specified number of layers in the backbone.

    Arguments:
        train_pretrained (bool): If True, returns a model with backbone pre-trained on Imagenet
    """

    backbone = mobilenet_v2(pretrained=is_main_process())
    return_layers = {"features": "0"}
    num_channels = 1280

    backbone = BackboneBase(backbone, return_layers, train_backbone, num_channels)

    return backbone


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
    bbox_reg = list()
    logits = list()

    in_channels = round(576 * width_mult)
    channels = [in_channels, 1280, 512, 256, 256, 64]

    for i in range(5):
        bbox_reg.append(SeperableConv2d(channels[i], cfg[i] * 4, 3, padding=1))
        logits.append(SeperableConv2d(channels[i], cfg[i] * num_classes, 3, padding=1))

    bbox_reg.append(nn.Conv2d(channels[-1], cfg[-1] * 4, 1))
    logits.append(nn.Conv2d(channels[-1], cfg[-1] * num_classes, 1))

    return bbox_reg, logits


model_urls = {'ssd_lite_mobilenet_v2': ''}


def build(args):
    if args.image_size != 300:
        raise NotImplementedError(
            "You specified image_size [{}]. However, currently only "
            "MobileNetV2SSD300 (image_size=300) is supported!".format(args.image_size),
        )

    backbone = build_backbone(train_backbone=True)
    extras_layers = build_extras(backbone.num_channels)

    anchor_nms_cfg = [6, 6, 6, 6, 6, 6]  # number of boxes per feature map location
    multibox_head = build_multibox(anchor_nms_cfg, args.num_classes)

    model = SSDLiteWithMobileNetV2(
        backbone,
        extras_layers,
        multibox_head,
        onnx_export=args.onnx_export,
        score_thresh=args.score_thresh,
        image_size=args.image_size,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        min_ratio=20,
        max_ratio=80,
        steps=[16, 32, 64, 100, 150, 300],
        clip=True,
        min_sizes=[60, 105, 150, 195, 240, 285],
        max_sizes=[105, 150, 195, 240, 285, 330],
    )

    return model
