# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Zhiqiang Wang (zhiqwang@outlook.com)

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import torchvision
from torchvision.ops.boxes import box_iou, batched_nms

from . import _utils as det_utils

from torch.jit.annotations import List, Optional, Dict, Tuple


@torch.jit.unused
def _onnx_get_num_priors(ob):
    # type: (Tensor) -> int
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[0].unsqueeze(0)

    return num_anchors


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


class MultiBoxLiteHead(nn.Module):
    """
    Adds a simple MultiBox Lite Head with classification and regression heads
    Arguments:
        hidden_dims (list): number of channels of the input feature
        num_anchors (list): number of anchors to be predicted
    """

    def __init__(self, hidden_dims, num_anchors, num_classes):
        super().__init__()

        self.cls_logits = nn.ModuleList()
        self.bbox_pred = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            self.cls_logits.append(SeperableConv2d(hidden_dims[i], num_anchors[i] * num_classes, 3, padding=1))
            self.bbox_pred.append(SeperableConv2d(hidden_dims[i], num_anchors[i] * 4, 3, padding=1))

        self.cls_logits.append(nn.Conv2d(hidden_dims[-1], num_anchors[-1] * num_classes, 1))
        self.bbox_pred.append(nn.Conv2d(hidden_dims[-1], num_anchors[-1] * 4, 1))

    def get_result_from_cls_logits(self, x, idx):
        # type: (Tensor, int) -> Tensor
        """
        This is equivalent to self.cls_logits[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.cls_logits:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.cls_logits:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_bbox_pred(self, x, idx):
        # type: (Tensor, int) -> Tensor
        """
        This is equivalent to self.bbox_pred[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = 0
        for m in self.bbox_pred:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.bbox_pred:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, features):
        # type: (List[Tensor]) -> Tuple[Tensor, Tensor]
        logits = []
        bbox_reg = []

        for i in range(len(features)):
            logits.append(self.get_result_from_cls_logits(features[i], i))
            bbox_reg.append(self.get_result_from_bbox_pred(features[i], i))

        logits, bbox_reg = concat_box_prediction_layers(logits, bbox_reg)

        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1)
    box_regression = torch.cat(box_regression_flattened, dim=1)
    return box_cls, box_regression


class SetCriterion(nn.Module):
    """
    Implements MultiBox based SSD Heads.
    Arguments:
        prior_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        multibox_head (nn.Module): module that computes the objectness and regression deltas
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'hard_negative_mining': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        prior_generator,
        multibox_head,
        variances,
        iou_thresh,
        negative_positive_ratio,
        score_thresh,
        nms_thresh,
        detections_per_img,
    ):
        super().__init__()
        self.prior_generator = prior_generator
        self.multibox_head = multibox_head
        self.box_coder = det_utils.BoxCoder(tuple(variances))

        # used during training
        self.iou_thresh = iou_thresh
        self.hard_negative_mining = det_utils.BalancedPositiveNegativeSampler(
            negative_positive_ratio,
        )

        # used during testing
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def forward(
        self,
        priors,  # type: Tensor
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        targets,  # type: List[Dict[str, Tensor]]
    ):
        # type: (...) -> Dict[str, Tensor]
        losses = {}

        regression_targets, labels = self.select_training_samples(priors, targets)
        loss_classifier, loss_box_reg = self.compute_loss(
            box_regression, class_logits, regression_targets, labels)

        losses = {
            'loss_box_reg': loss_box_reg,
            'loss_classifier': loss_classifier,
        }

        return losses

    def assign_targets_to_priors(self, gt_boxes, gt_labels, priors):
        # type: (List[Tensor], List[Tensor], Tensor) -> Tuple[List[Tensor], List[Tensor]]
        """Assign ground truth boxes and targets to priors.
        Args:
            gt_boxes (List[Tensor]): [num_targets, 4]: ground truth boxes
            gt_labels (List[Tensor]): [num_targets,]: labels of targets
            priors (Tensor): [num_priors, 4]: XYXY_REL BoxMode
        Returns:
            boxes (List[Tensor]): [num_priors, 4] real values for priors.
            labels (List[Tensor]): [num_priros] labels for priors.
        """
        boxes = []
        labels = []
        for gt_boxes_in_image, gt_labels_in_image in zip(gt_boxes, gt_labels):

            match_quality_matrix = box_iou(gt_boxes_in_image, priors)  # num_targets x num_priors
            if match_quality_matrix.numel() == 0:
                # empty targets or proposals not supported during training
                if match_quality_matrix.shape[0] == 0:
                    raise ValueError(
                        "No ground-truth boxes available for one of the images "
                        "during training")
                else:
                    raise ValueError(
                        "No default boxes available for one of the images "
                        "during training")
            matched_vals, matches = match_quality_matrix.max(0)  # num_priors
            _, best_prior_per_target_index = match_quality_matrix.max(1)  # num_targets

            for target_index, prior_index in enumerate(best_prior_per_target_index):
                matches[prior_index] = target_index
            # 2.0 is used to make sure every target has a prior assigned
            matched_vals.index_fill_(0, best_prior_per_target_index, 2)

            labels_in_image = gt_labels_in_image[matches]  # num_priors
            labels_in_image[matched_vals < self.iou_thresh] = 0  # the backgound id
            boxes_in_image = gt_boxes_in_image[matches]

            boxes.append(boxes_in_image)
            labels.append(labels_in_image)
        return boxes, labels

    def select_training_samples(self, priors, targets):
        # type: (Tensor, Optional[List[Dict[str, Tensor]]]) -> Tuple[Tensor, Tensor]
        assert targets is not None
        dtype = priors.dtype

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        priors_xyxy = det_utils.xywha_to_xyxy(priors)
        # get boxes indices for each priors
        boxes, labels = self.assign_targets_to_priors(gt_boxes, gt_labels, priors_xyxy)

        gt_locations = []
        for img_id in range(len(targets)):
            locations = self.box_coder.encode(boxes[img_id], priors)
            gt_locations.append(locations)

        regression_targets = torch.stack(gt_locations, 0)
        labels = torch.stack(labels, 0)

        return regression_targets, labels

    def compute_loss(self, box_regression, class_logits, regression_targets, labels):
        """Implement SSD MultiBox Loss. Basically, MultiBox loss combines
            objectness loss and Smooth L1 regression loss.
        Args:
            box_regression (Tensor): [batch_size, num_priors, 4] Predicted locations.
            class_logits (Tensor): [batch_size, num_priors, num_classes] Class predictions.
            regression_targets (Tensor): [batch_size, num_priors, 4] Real boxes corresponding all the priors.
            labels (Tensor): [batch_size, num_priors] Real labels of all the priors.
        """
        num_classes = class_logits.shape[2]
        with torch.no_grad():
            loss = - F.log_softmax(class_logits, dim=2)[:, :, 0]
            mask = self.hard_negative_mining(loss, labels)

        class_logits = class_logits[mask, :]

        objectness_loss = F.cross_entropy(
            class_logits.reshape(-1, num_classes),
            labels[mask],
            reduction='sum',
        )

        pos_mask = labels > 0
        box_regression = box_regression[pos_mask, :].reshape(-1, 4)
        regression_targets = regression_targets[pos_mask, :].reshape(-1, 4)

        box_loss = F.smooth_l1_loss(
            box_regression,
            regression_targets,
            reduction='sum',
        )
        num_pos = regression_targets.shape[0]

        return objectness_loss / num_pos, box_loss / num_pos


class PostProcess(nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }
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
        # type: (Tensor) -> int
        if torchvision._is_tracing():
            num_anchors = _onnx_get_num_priors(priors)
        else:
            num_anchors = priors.shape[0]
        return num_anchors

    def forward(self, pred_logits: Tensor, pred_boxes: Tensor, priors: Tensor, target_sizes: Tensor):
        """ Perform the computation. At test time, postprocess_detections is the final layer of SSD.
        Decode location preds, apply non-maximum suppression to location predictions based on conf
        scores and threshold to a detections_per_img number of output predictions
        for both confidence score and locations.

        Parameters:
            pred_logits : [batch_size, num_priors, num_classes] class predictions.
            pred_boxes : [batch_size, num_priors, 4] predicted locations.
            priors : [num_priors, 4] real boxes corresponding all the priors.
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        device = pred_logits.device
        num_classes = pred_logits.shape[-1]
        num_priors = self._get_num_priors(priors)

        out_boxes = self.box_coder.decode(pred_boxes, priors)  # batch_size x num_priors x 4
        out_scores = F.softmax(pred_logits, -1)

        results = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        for boxes, scores, target_size in zip(out_boxes, out_scores, target_sizes):
            # For each class, perform nms
            boxes = boxes.reshape(num_priors, 1, 4)
            boxes = boxes.expand(num_priors, num_classes, 4)
            boxes = boxes * target_size.flip(0).repeat(2)

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

            results.append({'scores': scores, 'labels': labels, 'boxes': boxes})

        return results
