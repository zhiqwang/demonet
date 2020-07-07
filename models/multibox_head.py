import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision

from .prior_box import AnchorGenerator
from .prior_matcher import PriorMatcher, decode

from torch.jit.annotations import List, Dict, Tuple


class MultiBoxHeads(nn.Module):
    def __init__(
        self,
        variances=[0.1, 0.2],
        iou_threshold=0.5,
        background_label=0,
        neg_pos_ratio=3,
        score_thresh=0.5,
        nms_thresh=0.45,
        top_k=100,
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
        self.variances = variances
        self.background_label = background_label
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k

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
        self.build_matcher = PriorMatcher(variances, iou_threshold)

    def forward(
        self,
        loc,       # type: Tensor
        conf,      # type: Tensor
        features,  # type: List[Tensor]
        targets,   # type: List[Dict[str, Tensor]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        loc = loc.view(loc.shape[0], -1, 4)  # loc preds
        conf = conf.view(conf.shape[0], loc.shape[1], -1)  # conf preds
        priors = self.prior_generator(features)

        predictions = None
        losses = {}

        if self.training:
            assert targets is not None
            gt_loc, gt_labels = self.build_matcher(priors, targets)
            loss_objectness, loss_box_reg = multibox_loss(loc, conf, gt_loc, gt_labels)
            losses = {
                'loc': loss_box_reg,
                'conf': loss_objectness,
            }
        else:
            predictions = self.postprocess(loc, conf, priors)
        return predictions, losses

    def postprocess(self, loc_data, conf_data, priors):
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
            decoded_boxes = decode(loc_data[i], priors, self.variances)  # num_priors x 4
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

            keep = torchvision.ops.boxes.batched_nms(
                decoded_boxes, conf_scores, labels, self.nms_thresh,
            )

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


def multibox_loss(
    pred_loc, pred_conf, gt_loc, gt_labels,
    neg_pos_ratio=3,
):
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
        mask = hard_negative_mining(loss, gt_labels, neg_pos_ratio)

    pred_conf = pred_conf[mask, :]

    objectness_loss = F.cross_entropy(
        pred_conf.reshape(-1, num_classes),
        gt_labels[mask],
        reduction='sum',
    )

    pos_mask = gt_labels > 0
    pred_loc = pred_loc[pos_mask, :].reshape(-1, 4)
    gt_loc = gt_loc[pos_mask, :].reshape(-1, 4)

    smooth_l1_loss = F.smooth_l1_loss(
        pred_loc,
        gt_loc,
        reduction='sum',
    )
    num_pos = gt_loc.shape[0]

    return objectness_loss / num_pos, smooth_l1_loss / num_pos


def hard_negative_mining(loss, targets, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
        cut the number of negative predictions to make sure the ratio
        between the negative examples and positive examples is no more
        the given ratio for an image.
    Args:
        loss (batch_size, num_priors): the loss for each example.
        targets (batch_size, num_priors): the targets.
        neg_pos_ratio: the ratio between the negative examples and positive examples.
    """
    pos_mask = targets > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = - math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask
