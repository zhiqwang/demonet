# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.
"""
Implements the Generalized SSD framework
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import torchvision
from torchvision.ops.boxes import batched_nms

from . import _utils as det_utils

from util.misc import NestedTensor, nested_tensor_from_tensor_list

from torch.jit.annotations import Tuple, List, Dict


class GeneralizedSSD(nn.Module):
    """
    Main class for Generalized SSD.

    Arguments:
        backbone (nn.Module):
        multibox_heads (nn.Module): takes the features + the proposals from the multibox and computes
            detections from it.
    """

    def __init__(self, backbone, prior_generator, multibox_head):
        super().__init__()
        self.backbone = backbone
        self.prior_generator = prior_generator
        self.multibox_head = multibox_head

    def forward(self, samples):
        # type: (NestedTensor) -> Dict[str, Tensor]
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

        out = {'pred_logits': logits, 'pred_boxes': bbox_reg, 'priors': priors}

        return out


@torch.jit.unused
def _onnx_get_num_priors(ob):
    # type: (Tensor, int) -> int
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[0].unsqueeze(0)

    return num_anchors


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(
        self,
        variances=(0.1, 0.2),
        score_thresh=0.5,
        nms_thresh=0.45,
        detections_per_img=100,
    ):
        super().__init__()
        self.box_coder = det_utils.BoxCoder(variances)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def _get_num_priors(self, priors):
        # type: (Tensor) -> Tensor
        if torchvision._is_tracing():
            num_anchors = _onnx_get_num_priors(priors)
        else:
            num_anchors = priors.shape[0]
        return num_anchors

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        # type: (Dict[str, Tensor], Tensor) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """ Perform the computation. At test time, postprocess_detections is the final layer of SSD.
        Decode location preds, apply non-maximum suppression to location predictions based on conf
        scores and threshold to a detections_per_img number of output predictions
        for both confidence score and locations.

        Parameters:
            outputs: raw outputs of the model, which consists of:
                - out_logits (Tensor): [batch_size, num_priors, num_classes] class predictions.
                - out_bbox (Tensor): [batch_size, num_priors, 4] predicted locations.
                - priors (Tensor): [num_priors, 4] real boxes corresponding all the priors.
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox, priors = outputs['pred_logits'], outputs['pred_boxes'], outputs['priors']

        device = out_logits.device
        num_classes = out_logits.shape[-1]
        num_priors = self._get_num_priors(priors)

        pred_boxes = self.box_coder.decode(out_bbox, priors)  # batch_size x num_priors x 4
        prob = F.softmax(out_logits, -1)

        results = []
        for boxes, scores, target_size in zip(pred_boxes, prob, target_sizes):
            # For each class, perform nms
            boxes = boxes.reshape(num_priors, 1, 4)
            boxes = boxes.expand(num_priors, num_classes, 4)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = det_utils.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per level
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            boxes = boxes * target_size.flip(0).repeat(2)

            results.append({'scores': scores, 'labels': labels, 'boxes': boxes})

        return results
