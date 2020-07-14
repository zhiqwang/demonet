# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Zhiqiang Wang (zhiqwang@outlook.com)
"""
Implements the Generalized SSD framework

Mostly copy-paste from https://github.com/pytorch/vision/blob/master/torchvision/models/detection/generalized_rcnn.py
"""

import torch
from torch import nn
import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor

from util.misc import NestedTensor, nested_tensor_from_tensor_list


class GeneralizedSSD(nn.Module):
    """
    Main class for Generalized R-CNN.
    Arguments:
        backbone (nn.Module):
        multibox_heads (nn.Module): takes the features + the proposals from the multibox and computes
            detections from it.
    """

    def __init__(self, backbone, multibox_heads):
        super().__init__()
        self.backbone = backbone
        self.multibox_heads = multibox_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, samples, targets=None):
        # type: (NestedTensor, Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
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

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("DEMONET always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (detector_losses, detections)
        else:
            return self.eager_outputs(detector_losses, detections)
