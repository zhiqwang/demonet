# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
"""
Implements the Generalized SSD framework
"""

import torch
from torch import nn, Tensor

from util.misc import NestedTensor, nested_tensor_from_tensor_list


class GeneralizedSSD(nn.Module):
    """
    Main class for Generalized SSD.

    Arguments:
        backbone (nn.Module):
        multibox_heads (nn.Module): takes the features + the proposals from the multibox and computes
            detections from it.
    """

    def __init__(self, backbone, prior_generator, multibox_head, post_process):
        super().__init__()
        self.backbone = backbone
        self.prior_generator = prior_generator
        self.multibox_head = multibox_head
        self.post_process = post_process

    def forward(self, samples: NestedTensor, target_sizes: Tensor):
        """
        Arguments:
            samples (NestedTensor): Expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features = self.backbone(samples)

        priors = self.prior_generator(features)  # BoxMode: XYWHA_REL
        logits, bbox_reg = self.multibox_head(features)

        out = self.post_process(logits, bbox_reg, priors, target_sizes)

        return out
