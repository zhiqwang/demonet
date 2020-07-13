import torch
from torch import nn, Tensor
import torch.nn.functional as F

import torchvision
from torchvision.ops.boxes import box_iou, batched_nms

from . import _utils as det_utils

from torch.jit.annotations import List, Optional, Dict, Tuple


@torch.jit.unused
def _onnx_get_num_priors(ob):
    # type: (Tensor, int) -> int
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


class MultiBox(nn.Module):
    """
    Implements MultiBox Heads.
    Arguments:
        prior_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'hard_negative_mining': det_utils.BalancedPositiveNegativeSampler,
        'post_nms_top_n': int,
    }

    def __init__(
        self,
        prior_generator,
        head,
        variances=[0.1, 0.2],
        iou_thresh=0.5,
        background_label=0,
        negative_positive_ratio=3,
        score_thresh=0.5,
        nms_thresh=0.45,
        post_nms_top_n=100,
    ):
        super().__init__()
        self.prior_generator = prior_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(tuple(variances))

        # used during training
        self.iou_thresh = iou_thresh
        self.hard_negative_mining = det_utils.BalancedPositiveNegativeSampler(
            negative_positive_ratio,
        )

        # used during testing
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.post_nms_top_n = post_nms_top_n

    def forward(
        self,
        features,  # type: List[Tensor]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        priors = self.prior_generator(features)  # BoxMode: XYWHA_REL
        objectness, pred_bbox_deltas = self.head(features)

        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

        gt_locations = []
        gt_labels = []
        predictions = {}
        losses = {}
        if self.training:
            assert targets is not None
            for target in targets:
                priors_xyxy = det_utils.xywha_to_xyxy(priors)
                box, label = target['boxes'], target['labels']
                box, label = self.assign_targets_to_priors(box, label, priors_xyxy)
                locations = self.box_coder.encode(box, priors)
                gt_locations.append(locations)
                gt_labels.append(label)

            gt_locations_batch = torch.stack(gt_locations, 0)
            gt_labels_batch = torch.stack(gt_labels, 0)
            loss_objectness, loss_box_reg = self.compute_loss(
                pred_bbox_deltas, objectness, gt_locations_batch, gt_labels_batch)

            losses = {
                'pred_bbox_deltas': loss_box_reg,
                'objectness': loss_objectness,
            }
        else:
            boxes, scores, labels = self.filter_priors(priors, pred_bbox_deltas, objectness)
            predictions = self.parse_predictions(boxes, scores, labels)
        return predictions, losses

    def assign_targets_to_priors(self, gt_boxes, gt_labels, priors):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
        """Assign ground truth boxes and targets to priors.
        Args:
            gt_boxes (Tensor): [num_targets, 4]: ground truth boxes
            gt_labels (Tensor): [num_targets,]: labels of targets
            priors (Tensor): [num_priors, 4]: XYXY_REL BoxMode
        Returns:
            boxes (Tensor): [num_priors, 4] real values for priors.
            labels (Tensor): [num_priros] labels for priors.
        """
        match_quality_matrix = box_iou(gt_boxes, priors)  # num_targets x num_priors
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")
        matched_vals, matches = match_quality_matrix.max(0)  # num_priors
        _, best_prior_per_target_index = match_quality_matrix.max(1)  # num_targets

        for target_index, prior_index in enumerate(best_prior_per_target_index):
            matches[prior_index] = target_index
        # 2.0 is used to make sure every target has a prior assigned
        matched_vals.index_fill_(0, best_prior_per_target_index, 2)

        labels = gt_labels[matches]  # num_priors
        labels[matched_vals < self.iou_thresh] = 0  # the backgound id
        boxes = gt_boxes[matches]
        return boxes, labels

    def _get_num_priors(self, priors):
        # type: (Tensor) -> Tensor
        if torchvision._is_tracing():
            num_anchors = _onnx_get_num_priors(priors)
        else:
            num_anchors = priors.shape[0]
        return num_anchors

    def filter_priors(self, priors, pred_bbox_deltas, objectness):
        # type: (Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
        """
        At test time, Detect is the final layer of SSD. Decode location preds,
        apply non-maximum suppression to location predictions based on conf
        scores and threshold to a post_nms_top_n number of output predictions
        for both confidence score and locations.

        Args:
            pred_bbox_deltas (tensor): [batch_size, num_priors, 4] predicted locations.
            objectness (tensor): [batch_size, num_priors, num_classes] class predictions.
            priors (tensor): [num_priors, 4] real boxes corresponding all the priors.
        """
        objectness = F.softmax(objectness, dim=2)
        num_priors = self._get_num_priors(priors)

        final_boxes = []
        final_scores = []
        final_labels = []
        for scores, bbox_reg in zip(objectness, pred_bbox_deltas):
            decoded_boxes = self.box_coder.decode(bbox_reg, priors)  # num_priors x 4
            # For each class, perform nms
            num_priors, num_classes = scores.shape

            decoded_boxes = decoded_boxes.reshape(num_priors, 1, 4)
            decoded_boxes = decoded_boxes.expand(num_priors, num_classes, 4)
            labels = torch.arange(num_classes, device=decoded_boxes.device)
            labels = labels.reshape(1, num_classes).expand_as(scores)

            # remove predictions with the background label
            decoded_boxes = decoded_boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            decoded_boxes = decoded_boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring decoded_boxes
            indices = torch.nonzero(scores > self.score_thresh, as_tuple=False).squeeze(1)
            decoded_boxes = decoded_boxes[indices]
            scores = scores[indices]
            labels = labels[indices]
            # non-maximum suppression, independently done per level
            keep = batched_nms(decoded_boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.post_nms_top_n]

            decoded_boxes, scores, labels = decoded_boxes[keep], scores[keep], labels[keep]
            final_boxes.append(decoded_boxes)
            final_scores.append(scores)
            final_labels.append(labels)

        return final_boxes, final_scores, final_labels

    def parse_predictions(self, boxes, scores, labels):
        # type: (Tensor, Tensor, Tensor) -> List[Dict[str, Tensor]]

        predictions = []

        for box, score, label in zip(boxes, scores, labels):
            prediction = {}

            # Get detections with confidence higher than threshold.
            top_indices = [i for i, conf in enumerate(score) if conf >= self.score_thresh]

            top_box = box[top_indices, :]
            top_score = score[top_indices]
            top_label = label[top_indices]

            prediction['boxes'] = torch.clamp(top_box, min=0.0, max=1.0)
            prediction['scores'] = top_score
            prediction['labels'] = top_label

            predictions.append(prediction)

        return predictions

    def compute_loss(self, pred_loc, pred_conf, gt_loc, gt_labels):
        """Implement SSD MultiBox Loss. Basically, MultiBox loss combines
            objectness loss and Smooth L1 regression loss.
        Args:
            pred_loc (Tensor): [batch_size, num_priors, 4] Predicted locations.
            pred_conf (Tensor): [batch_size, num_priors, num_classes] Class predictions.
            gt_loc (Tensor): [batch_size, num_priors, 4] Real boxes corresponding all the priors.
            gt_labels (Tensor): [batch_size, num_priors] Real gt_labels of all the priors.
        """
        num_classes = pred_conf.shape[2]
        with torch.no_grad():
            loss = - F.log_softmax(pred_conf, dim=2)[:, :, 0]
            mask = self.hard_negative_mining(loss, gt_labels)

        pred_conf = pred_conf[mask, :]

        objectness_loss = F.cross_entropy(
            pred_conf.reshape(-1, num_classes),
            gt_labels[mask],
            reduction='sum',
        )

        pos_mask = gt_labels > 0
        pred_loc = pred_loc[pos_mask, :].reshape(-1, 4)
        gt_loc = gt_loc[pos_mask, :].reshape(-1, 4)

        box_loss = F.smooth_l1_loss(
            pred_loc,
            gt_loc,
            reduction='sum',
        )
        num_pos = gt_loc.shape[0]

        return objectness_loss / num_pos, box_loss / num_pos
