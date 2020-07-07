import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchvision.ops.boxes import box_iou, batched_nms
from . import _utils as det_utils

from torch.jit.annotations import Tuple, List, Dict, Optional


class PriorMatcher(nn.Module):
    """This class computes an assignment between the priors and the targets of the network.
    """
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
    }

    def __init__(self, prior_generator, variances, iou_threshold):
        super().__init__()
        self.prior_generator = prior_generator
        self.box_coder = det_utils.BoxCoder(variances=variances)
        self.iou_threshold = iou_threshold

    def forward(self, features, loc, conf, targets=None):
        # type: (List[Tensor], Tensor, Tensor, Optional[List[Dict[str, Tensor]]]) -> Tuple[Tensor, Tensor]
        priors_xywha = self.prior_generator(features)
        priors_xyxy = det_utils.xywha_to_xyxy(priors_xywha)
        gt_locations = torch.jit.annotate(List[Tensor], [])
        gt_labels = torch.jit.annotate(List[Tensor], [])
        predictions = None
        if self.training:
            assert targets is not None
            for target in targets:
                box, label = target['boxes'], target['labels']
                box, label = self.assign_targets_to_priors(box, label, priors_xyxy)
                locations = self.box_coder.encode(box, priors_xywha)
                gt_locations.append(locations)
                gt_labels.append(label)

            gt_locations = torch.stack(gt_locations, 0)
            gt_labels = torch.stack(gt_labels, 0)
        else:
            predictions = self.filter_proposals(loc, conf, priors_xywha)
        return gt_locations, gt_labels, predictions

    def assign_targets_to_priors(self, gt_boxes, gt_labels, priors):
        # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]
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
        labels[matched_vals < self.iou_threshold] = 0  # the backgound id
        boxes = gt_boxes[matches]
        return boxes, labels

    def filter_proposals(self, loc_data, conf_data, priors):
        """
        At test time, Detect is the final layer of SSD. Decode location preds,
        apply non-maximum suppression to location predictions based on conf
        scores and threshold to a top_k number of output predictions for both
        confidence score and locations.

        Args:
            loc_data (tensor): [batch_size, num_priors x 4] predicted locations.
            conf_data (tensor): [batch_size, num_priors, num_classes] class predictions.
            priors (tensor): [num_priors, 4] real boxes corresponding all the priors.
        """
        conf_data = F.softmax(conf_data, dim=2)
        predictions = torch.jit.annotate(List[Dict[str, Tensor]], [])
        batch_size = loc_data.shape[0]
        num_priors = priors.shape[0]
        # conf_preds: batch_size x num_priors x num_classes
        conf_preds = conf_data.view(batch_size, num_priors, -1)

        # Decode predictions into bboxes.
        for i in range(batch_size):
            decoded_boxes = self.box_coder.decode(loc_data[i], priors)  # num_priors x 4
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()  # num_priors x num_classes

            num_priors, num_classes = conf_scores.shape

            decoded_boxes = decoded_boxes.view(num_priors, 1, 4)
            decoded_boxes = decoded_boxes.expand(num_priors, num_classes, 4)
            labels = torch.arange(num_classes, device=decoded_boxes.device)
            labels = labels.view(1, num_classes).expand_as(conf_scores)

            # remove predictions with the background label
            decoded_boxes = decoded_boxes[:, 1:]
            conf_scores = conf_scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            decoded_boxes = decoded_boxes.reshape(-1, 4)
            conf_scores = conf_scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring decoded_boxes
            indices = torch.nonzero(conf_scores > self.score_thresh, as_tuple=False).squeeze(1)
            decoded_boxes = decoded_boxes[indices]
            conf_scores = conf_scores[indices]
            labels = labels[indices]

            keep = batched_nms(decoded_boxes, conf_scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            keep = keep[:self.top_k]

            decoded_boxes = decoded_boxes[keep]
            conf_scores = conf_scores[keep]
            labels = labels[keep]

            prediction = self.parse(labels, conf_scores, decoded_boxes)
            predictions.append(prediction)

        return predictions

    def parse(self, det_label, det_conf, det_boxes):
        # Parse the outputs
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
