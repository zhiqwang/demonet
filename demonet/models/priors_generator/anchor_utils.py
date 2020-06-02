import torch
import torch.nn as nn

from torchvision.ops.boxes import box_iou

from ..box_heads.box_utils import (
    xywha_to_xyxy,
    xyxy_to_xywha,
)


class PriorMatcher(nn.Module):
    """This class computes an assignment between the priors and the targets of the network.
    """
    def __init__(self, variances, iou_threshold):
        super().__init__()
        self.variances = variances
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def forward(self, priors_xywha, targets):

        priors_xyxy = xywha_to_xyxy(priors_xywha)
        gt_locations = list()
        gt_labels = list()
        for target in targets:
            box, label = target['boxes'], target['labels']
            box, label = assign_targets_to_priors(box, label, priors_xyxy, self.iou_threshold)
            locations = encode(box, priors_xywha, self.variances)
            gt_locations.append(locations)
            gt_labels.append(label)

        gt_locations = torch.stack(gt_locations, 0)
        gt_labels = torch.stack(gt_labels, 0)
        return gt_locations, gt_labels


def assign_targets_to_priors(gt_boxes, gt_labels, priors, iou_threshold):
    """Assign ground truth boxes and targets to priors.
    Args:
        gt_boxes (Tensor): [num_targets, 4]: ground truth boxes
        gt_labels (Tensor): [num_targets,]: labels of targets
        priors (Tensor): [num_priors, 4]: XYXY_REL BoxMode
        iou_threshold (float): iou threshold
    Returns:
        boxes (Tensor): [num_priors, 4] real values for priors.
        labels (Tensor): [num_priros] labels for priors.
    """
    match_quality_matrix = box_iou(gt_boxes, priors)  # num_targets x num_priors
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
    labels[matched_vals < iou_threshold] = 0  # the backgound id
    boxes = gt_boxes[matches]
    return boxes, labels


def encode(boxes, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have boxes (based on jaccard overlap) with the prior boxes.
    Args:
        boxes (tensor): [num_priors, 4] coords of ground truth for each prior in XYXY_REL BoxMode
        priors (tensor): [num_priors, 4] prior boxes in XYWHA_REL BoxMode
        variances (list[float]): variances of prior boxes
    Return:
        encoded boxes (tensor) [num_priors, 4]
    """
    boxes = xyxy_to_xywha(boxes)
    locations = boxes_to_locations(boxes, priors, variances)
    return locations


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(locations, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        locations (tensor): [num_priors, 4] location predictions for locations layers
        priors (tensor): [num_priors, 4] prior boxes in XYWHA_REL BoxMode
        variances (list[float]): variances of prior boxes
    Return:
        decoded bounding box predictions
    """

    boxes = locations_to_boxes(locations, priors, variances)
    boxes = xywha_to_xyxy(boxes)
    return boxes


def boxes_to_locations(boxes, priors, variances):
    r"""Convert boxes into regressional location results of SSD
    Args:
        boxes (Tensor): [num_targets, 4] in XYWHA_REL BoxMode
        priors (Tensor): [num_targets] in XYWHA_REL BoxMode
        variances (list(float)): used to change the scale of center.
            change of scale of size.
    """
    # priors can have one dimension less
    if priors.dim() + 1 == boxes.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        (boxes[..., :2] - priors[..., :2]) / priors[..., 2:] / variances[0],
        torch.log(boxes[..., 2:] / priors[..., 2:]) / variances[1],
    ], dim=boxes.dim() - 1)


def locations_to_boxes(locations, priors, variances):
    r"""Convert regressional location results of SSD into boxes in XYWHA_REL BoxMode
    The conversion:
        $$predicted\_center \times variance_center =
            \frac{real\_center - prior\_center}{prior\_hw}$$
        $$\exp(predicted\_hw \times variance_size) = \frac{real\_hw}{prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (Tensor): [batch_size, num_priors, 4] the regression output of SSD.
            It will contain the outputs as well.
        priors (Tensor): [num_priors, 4] or [batch_size, num_priors, 4] prior boxes.
        variances (list(float)): used to change the scale of center and
            change of scale of size.
    Returns:
        boxes: priors (Tensor): converted XYWHA_REL BoxMode. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * variances[0] * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * variances[1]) * priors[..., 2:],
    ], dim=locations.dim() - 1)
