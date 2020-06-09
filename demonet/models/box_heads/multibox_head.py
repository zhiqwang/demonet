import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..priors_generator.prior_box import PriorBoxGenerator
from ..priors_generator.anchor_utils import PriorMatcher, decode

from .box_utils import hard_negative_mining


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
        # prior box
        image_size=300,
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        feature_maps=[38, 19, 10, 5, 3, 1],
        min_ratio=20,
        max_ratio=90,
        steps=[8, 16, 32, 64, 100, 300],
        clip=False,
        min_sizes=None,
        max_sizes=None,
    ):
        super().__init__()
        self.variances = variances
        self.background_label = background_label
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.top_k = top_k

        self.image_size = image_size
        self.aspect_ratios = aspect_ratios
        self.feature_maps = feature_maps
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.steps = steps
        self.clip = clip
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes

        self.build_priors = self.prior_generator()
        self.build_matcher = PriorMatcher(variances, iou_threshold)

    def forward(self, loc, conf, targets):
        loc = loc.view(loc.shape[0], -1, 4)  # loc preds
        conf = conf.view(conf.shape[0], loc.shape[1], -1)  # conf preds
        priors = self.build_priors().to(loc.device)

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

    def prior_generator(self):

        priors = PriorBoxGenerator(
            image_size=self.image_size,
            aspect_ratios=self.aspect_ratios,
            feature_maps=self.feature_maps,
            min_ratio=self.min_ratio,
            max_ratio=self.max_ratio,
            steps=self.steps,
            clip=self.clip,
            min_sizes=self.min_sizes,
            max_sizes=self.max_sizes,
        )

        return priors

    @torch.no_grad()
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
        predictions = []
        batch_size = loc_data.shape[0]
        num_priors = priors.shape[0]
        # conf_preds: batch_size x num_priors x num_classes
        conf_preds = conf_data.view(batch_size, num_priors, -1)
        device = conf_data.device
        priors = priors.to(device)

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
            conf_scores = conf_scores[keep][:, None]
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
        prediction['labels'] = np.array([t.item() for t in top_label], dtype=np.int64)
        prediction['scores'] = np.array([t.item() for t in top_conf], dtype=np.float32)
        prediction['boxes'] = top_boxes.cpu().numpy().tolist()

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
