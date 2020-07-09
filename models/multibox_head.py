import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchvision.ops.boxes import box_iou, batched_nms

from . import _utils as det_utils
from .prior_box import AnchorGenerator

from torch.jit.annotations import List, Optional, Dict, Tuple


class MultiBoxHeads(nn.Module):
    """
    Implements MultiBox Heads.
    Arguments:
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'hard_negative_mining': det_utils.BalancedPositiveNegativeSampler,
        'post_nms_top_n': int,
    }

    def __init__(
        self,
        variances=[0.1, 0.2],
        iou_thresh=0.5,
        background_label=0,
        negative_positive_ratio=3,
        score_thresh=0.5,
        nms_thresh=0.45,
        post_nms_top_n=100,
        # parameter for prior box generator
        image_size=300,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        min_ratio=20,
        max_ratio=80,
        steps=[16, 32, 64, 100, 150, 300],
        clip=False,
        min_sizes=[60, 105, 150, 195, 240, 285],
        max_sizes=[105, 150, 195, 240, 285, 330],
    ):
        super().__init__()
        self.prior_generator = AnchorGenerator(
            image_size=image_size,
            aspect_ratios=aspect_ratios,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            steps=steps,
            clip=clip,
            min_sizes=min_sizes,
            max_sizes=max_sizes,
        )
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
        pred_bbox_deltas,  # type: Tensor
        objectness,  # type: Tensor
        features,  # type: List[Tensor]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        pred_bbox_deltas = pred_bbox_deltas.reshape(pred_bbox_deltas.shape[0], -1, 4)
        objectness = objectness.reshape(objectness.shape[0], pred_bbox_deltas.shape[1], -1)

        priors = self.prior_generator(features)  # BoxMode: XYWHA_REL
        priors_xyxy = det_utils.xywha_to_xyxy(priors)
        gt_locations = torch.jit.annotate(List[Tensor], [])
        gt_labels = torch.jit.annotate(List[Tensor], [])
        predictions = torch.jit.annotate(List[Dict[str, Tensor]], [])
        losses = {}
        if self.training:
            assert targets is not None
            for target in targets:
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
            predictions = self.filter_priors(priors, pred_bbox_deltas, objectness)
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

    def filter_priors(self, priors, pred_bbox_deltas, objectness):
        # type: (Tensor, Tensor, Tensor) -> List[Dict[str, Tensor]]
        """
        At test time, Detect is the final layer of SSD. Decode location preds,
        apply non-maximum suppression to location predictions based on conf
        scores and threshold to a post_nms_top_n number of output predictions
        for both confidence score and locations.

        Args:
            pred_bbox_deltas (tensor): [batch_size, num_priors x 4] predicted locations.
            objectness (tensor): [batch_size, num_priors, num_classes] class predictions.
            priors (tensor): [num_priors, 4] real boxes corresponding all the priors.
        """
        objectness = F.softmax(objectness, dim=2)
        predictions = torch.jit.annotate(List[Dict[str, Tensor]], [])
        num_images = pred_bbox_deltas.shape[0]
        # device = pred_bbox_deltas.device
        num_priors = priors.shape[0]
        # objectness: batch_size x num_priors x num_classes
        objectness = objectness.view(num_images, num_priors, -1)

        # Decode predictions into bboxes.
        for i in range(num_images):
            decoded_boxes = self.box_coder.decode(pred_bbox_deltas[i], priors)  # num_priors x 4
            # For each class, perform nms
            scores = objectness[i].clone()  # num_priors x num_classes
            num_priors, num_classes = scores.shape

            decoded_boxes = decoded_boxes.view(num_priors, 1, 4)
            decoded_boxes = decoded_boxes.expand(num_priors, num_classes, 4)
            labels = torch.arange(num_classes, device=decoded_boxes.device)
            labels = labels.view(1, num_classes).expand_as(scores)

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

            decoded_boxes = decoded_boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            prediction = self.parse(labels, scores, decoded_boxes)
            predictions.append(prediction)

        return predictions

    def parse(self, det_label, det_conf, det_boxes):
        # type: (Tensor, Tensor, Tensor) -> Dict[str, Tensor]
        prediction = {}

        # Get detections with confidence higher than threshold.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.score_thresh]

        top_label = det_label[top_indices]
        top_conf = det_conf[top_indices]
        top_boxes = det_boxes[top_indices, :]

        prediction = {}

        prediction['labels'] = top_label
        prediction['scores'] = top_conf
        prediction['boxes'] = torch.clamp(top_boxes, min=0.0, max=1.0)

        return prediction

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
