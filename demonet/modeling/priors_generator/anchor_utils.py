import math
import torch

from ..box_heads.box_utils import (
    pairwise_iou,
    xywha_to_xyxy,
    xyxy_to_xywha,
)


def get_min_max_sizes(
    min_ratio,
    max_ratio,
    input_size,
    mbox_source_num,
):
    step = int(math.floor(max_ratio - min_ratio) / (mbox_source_num - 2))
    min_sizes = list()
    max_sizes = list()
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(input_size * ratio / 100)
        max_sizes.append(input_size * (ratio + step) / 100)

    if min_ratio == 20:
        min_sizes = [input_size * 10 / 100.] + min_sizes
        max_sizes = [input_size * 20 / 100.] + max_sizes
    else:
        min_sizes = [input_size * 7 / 100.] + min_sizes
        max_sizes = [input_size * 15 / 100.] + max_sizes

    return min_sizes, max_sizes


def config_parse(config):
    cfg = dict()
    cfg['feature_maps'] = config.anchor_config.feature_maps
    cfg['min_dim'] = config.input_size
    cfg['steps'] = config.anchor_config.steps
    cfg['min_sizes'], cfg['max_sizes'] = get_min_max_sizes(
        config.anchor_config.min_ratio,
        config.anchor_config.max_ratio,
        config.input_size,
        len(cfg['feature_maps']),
    )
    cfg['aspect_ratios'] = config.anchor_config.aspect_ratios
    cfg['variance'] = [0.1, 0.2]
    cfg['clip'] = True
    return cfg


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


def assign_priors(gt_boxes, gt_labels, priors, iou_threshold):
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
    ious = pairwise_iou(gt_boxes.unsqueeze(0), priors.unsqueeze(1))  # num_priors x num_targets
    best_target_per_prior, best_target_per_prior_index = ious.max(1)  # num_priors
    best_prior_per_target, best_prior_per_target_index = ious.max(0)  # num_targets

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)

    labels = gt_labels[best_target_per_prior_index]  # num_priors
    labels[best_target_per_prior < iou_threshold] = 0  # the backgound id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels
