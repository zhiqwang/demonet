import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_utils import hard_negative_mining


class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss. Basically, MultiBox loss combines
            classification loss and Smooth L1 regression loss.
        """
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, pred_loc, pred_conf, gt_loc, gt_labels):
        """Compute classification loss and smooth l1 loss.
        Args:
            pred_loc (Tensor): [batch_size, num_priors, 4] Predicted locations.
            pred_conf (Tensor): [batch_size, num_priors, num_classes] Class predictions.
            gt_loc (Tensor): [batch_size, num_priors, 4] Real boxes corresponding all the priors.
            gt_labels (Tensor): [batch_size, num_priors] Real gt_labels of all the priors.
        """
        num_classes = pred_conf.shape[2]
        with torch.no_grad():
            loss = - F.log_softmax(pred_conf, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)

        pred_conf = pred_conf[mask, :]

        classification_loss = F.cross_entropy(
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

        losses = {
            'loc': smooth_l1_loss / num_pos,
            'conf': classification_loss / num_pos,
        }
        return losses
