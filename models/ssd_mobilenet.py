import warnings

import torch
from torch import nn, Tensor

from util.misc import NestedTensor, nested_tensor_from_tensor_list

from .backbone import build_backbone
from .multibox_head import MultiBox

from torch.jit.annotations import List, Optional, Dict, Tuple


class SSDLiteWithMobileNetV2(nn.Module):
    r"""MobileNet V2 SSD model class
    Args:
        backbone: MobileNet V2 backbone layers with extra layers
        head: "multibox head" consists of box_regression and class_logits conv layers
    """
    def __init__(
        self,
        backbone,
        score_thresh=0.3,
        image_size=300,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        min_ratio=20,
        max_ratio=80,
        steps=[16, 32, 64, 100, 150, 300],
        clip=True,
        min_sizes=[60, 105, 150, 195, 240, 285],
        max_sizes=[105, 150, 195, 240, 285, 330],
    ):
        super().__init__()

        self.backbone = backbone
        self.multibox_heads = MultiBox(
            score_thresh,
            image_size, aspect_ratios, min_ratio, max_ratio, steps,
            clip, min_sizes, max_sizes,
        )
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
        samples,  # type: NestedTensor
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            samples (NestedTensor): Expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features = self.backbone(samples)

        detections, detector_losses = self.multibox_heads(features, targets=targets)

        return self.eager_outputs(detector_losses, detections)
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("DEMONET always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (detector_losses, detections)
        else:
            return self.eager_outputs(detector_losses, detections)


model_urls = {'ssd_lite_mobilenet_v2': ''}


def build(args):
    if args.image_size != 300:
        raise NotImplementedError(
            "You specified image_size [{}]. However, currently only "
            "MobileNetV2SSD300 (image_size=300) is supported!".format(args.image_size),
        )

    backbone = build_backbone(train_backbone=True)

    model = SSDLiteWithMobileNetV2(
        backbone,
        score_thresh=args.score_thresh,
        image_size=args.image_size,
    )

    return model
