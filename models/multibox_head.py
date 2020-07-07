import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .prior_box import AnchorGenerator
from .prior_matcher import PriorMatcher

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

        prior_generator = AnchorGenerator(
            image_size=image_size,
            aspect_ratios=aspect_ratios,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            steps=steps,
            clip=clip,
            min_sizes=min_sizes,
            max_sizes=max_sizes,
        )
        self.build_matcher = PriorMatcher(prior_generator, variances, iou_threshold)

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

        predictions = None
        losses = {}

        gt_loc, gt_labels, predictions = self.build_matcher(features, loc, conf, targets)
        loss_objectness, loss_box_reg = multibox_loss(loc, conf, gt_loc, gt_labels)
        losses = {
            'loc': loss_box_reg,
            'conf': loss_objectness,
        }

        return predictions, losses


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
