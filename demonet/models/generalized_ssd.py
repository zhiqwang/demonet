# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
"""
Implements the Generalized SSD framework
"""
import warnings

import torch
from torch import nn, Tensor

from typing import List, Dict, Optional, Tuple


class GeneralizedSSD(nn.Module):
    """
    Main class for Generalized SSD.

    Arguments:
        backbone (nn.Module):
        multibox_heads (nn.Module): takes the features + the proposals from the multibox and computes
            detections from it.
    """

    def __init__(
        self,
        backbone: nn.Module,
        prior_generator: nn.Module,
        multibox_head: nn.Module,
        post_process: Optional[nn.Module],
        transform: Optional[nn.Module],
    ):
        super().__init__()
        self.backbone = backbone
        self.prior_generator = prior_generator
        self.multibox_head = multibox_head
        self.post_process = post_process
        self.transform = transform
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses: Dict[str, Tensor], detections: List[Dict[str, Tensor]]):
        if self.training:
            return losses

        return detections

    def forward(self, inputs: List[Tensor]):
        """
        Args:
            inputs (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []

        if not self.training:
            for img in inputs:
                val = img.shape[-2:]
                assert len(val) == 2
                original_image_sizes.append((val[0], val[1]))
        # Transform the input
        samples, _ = self.transform(inputs, None)
        # Compute the features
        features = self.backbone(samples.tensors)

        priors = self.prior_generator(features)  # BoxMode: XYWHA_REL
        logits, bbox_reg = self.multibox_head(features)
        out_ssd = {}
        detections = torch.jit.annotate(List[Dict[str, Tensor]], [])

        if self.training:
            out_ssd = {'pred_logits': logits, 'pred_boxes': bbox_reg, 'priors': priors}
        else:
            detections = self.post_process(logits, bbox_reg, priors)
            detections = self.transform.postprocess(detections, samples.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("DEMONET always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (out_ssd, detections)
        else:
            return self.eager_outputs(out_ssd, detections)
