import math

import torch
from torch import Tensor

from torch.jit.annotations import Tuple


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, negative_positive_ratio):
        # type: (int) -> None
        """
        Arguments:
            negative_positive_ratio (int): the ratio between the negative examples and
                positive examples.
        """
        self.negative_positive_ratio = negative_positive_ratio

    def __call__(self, loss, targets):
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
        """
        pos_mask = targets > 0
        num_pos = pos_mask.long().sum(dim=1, keepdim=True)
        num_neg = num_pos * self.negative_positive_ratio

        loss[pos_mask] = - math.inf
        _, indexes = loss.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        neg_mask = orders < num_neg
        return pos_mask | neg_mask


@torch.jit._script_if_tracing
def boxes_to_locations(boxes, priors, variances):
    # type: (Tensor, Tensor, Tuple[float]) -> Tensor
    r"""Convert boxes into regressional location results of SSD
    Args:
        boxes (Tensor): [num_targets, 4] in XYWHA_REL BoxMode
        priors (Tensor): [num_targets] in XYWHA_REL BoxMode
        variances (Tuple[float]): used to change the scale of center.
            change of scale of size.
    """
    # priors can have one dimension less
    if priors.dim() + 1 == boxes.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        (boxes[..., :2] - priors[..., :2]) / priors[..., 2:] / variances[0],
        torch.log(boxes[..., 2:] / priors[..., 2:]) / variances[1],
    ], dim=boxes.dim() - 1)


@torch.jit._script_if_tracing
def locations_to_boxes(locations, priors, variances):
    # type: (Tensor, Tensor, Tuple[float, float]) -> Tensor
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
        variances (Tuple[float]): used to change the scale of center and
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


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, variances):
        # type: (Tuple[float, float]) -> None
        """
        Arguments:
            variances (2-element tuple): variances of prior boxes
        """
        self.variances = variances

    def encode(self, boxes, priors):
        # type: (Tensor, Tensor) -> Tensor
        """Encode the variances from the priorbox layers into the ground truth boxes
        we have boxes (based on jaccard overlap) with the prior boxes.
        Args:
            boxes (tensor): [num_priors, 4] coords of ground truth for each prior in XYXY_REL BoxMode
            priors (tensor): [num_priors, 4] prior boxes in XYWHA_REL BoxMode
        Return:
            encoded boxes (tensor) [num_priors, 4]
        """
        boxes = xyxy_to_xywha(boxes)
        locations = boxes_to_locations(boxes, priors, self.variances)
        return locations

    def decode(self, locations, priors):
        # type: (Tensor, Tensor) -> Tensor
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            locations (tensor): [num_priors, 4] location predictions for locations layers
            priors (tensor): [num_priors, 4] prior boxes in XYWHA_REL BoxMode
        Return:
            decoded bounding box predictions
        """

        boxes = locations_to_boxes(locations, priors, self.variances)
        boxes = xywha_to_xyxy(boxes)
        return boxes


def xywha_to_xyxy(boxes):
    """ Convert BoxMode of boxes from XYWHA_REL to XYXY_REL.
    Args:
        boxes (Tensor): XYWHA_REL BoxMode
            default BoxMode from priorbox generator layers.
    Return:
        boxes (Tensor): XYXY_REL BoxMode
    """
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def xyxy_to_xywha(boxes):
    """ Convert BoxMode of boxes from XYXY_REL to XYWHA_REL.
    Args:
        boxes (Tensor): XYXY_REL BoxMode
    Return:
        boxes (Tensor): XYWHA_REL BoxMode
    """
    x0, y0, x1, y1 = boxes.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)
