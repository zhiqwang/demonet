import cv2
import matplotlib
import numpy as np

import torch
import torch.nn as nn

from demonet.modeling.utils import build_targets, bbox_iou

matplotlib.rc('font', **{'size': 11})

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def compute_loss(preds, targets, model):  # predictions, targets, model
    ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
    lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
    tcls, tbox, indices, anchor_vec = build_targets(model, targets)
    param = model.param  # hyperparameters
    arch = model.arch  # # (default, uCE, uBCE) detection architectures

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([param['cls_pw']]))
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([param['obj_pw']]))
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()  # weight=model.class_weights

    if 'F' in arch:  # add focal loss
        g = param['fl_gamma']
        BCEcls, BCEobj, BCE, CE = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g), FocalLoss(BCE, g), FocalLoss(CE, g)

    # Compute losses
    for i, pred_layer in enumerate(preds):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pred_layer[..., 0])  # target obj

        # Compute losses
        nb = len(b)  # number of targets in this layer
        if nb > 0:
            pred_subset = pred_layer[b, a, gj, gi]  # prediction subset corresponding to targets
            tobj[b, a, gj, gi] = 1.0  # obj
            # pred_subset[:, 2:4] = torch.sigmoid(pred_subset[:, 2:4])  # wh power loss (uncomment)

            # GIoU
            pxy = torch.sigmoid(pred_subset[:, 0:2])  # pxy = pxy * s - (s - 1) / 2,  s = 1.5  (scale_xy)
            pbox = torch.cat((pxy, torch.exp(pred_subset[:, 2:4]) * anchor_vec[i]), 1)  # predicted box
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou computation
            lbox += (1.0 - giou).mean()  # giou loss

            if 'default' in arch and model.nc > 1:  # cls loss (only if multiple classes)
                t = torch.zeros_like(pred_subset[:, 5:])  # targets
                t[range(nb), tcls[i]] = 1.0
                lcls += BCEcls(pred_subset[:, 5:], t)  # BCE
                # lcls += CE(pred_subset[:, 5:], tcls[i])  # CE

                # Instance-class weighting (use with reduction='none')
                # nt = t.sum(0) + 1  # number of targets per class
                # lcls += (BCEcls(pred_subset[:, 5:], t) / nt).mean() * nt.mean()  # v1
                # lcls += (BCEcls(pred_subset[:, 5:], t) / nt[tcls[i]].view(-1,1)).mean() * nt.mean()  # v2

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        if 'default' in arch:  # separate obj and cls
            lobj += BCEobj(pred_layer[..., 4], tobj)  # obj loss

        elif 'BCE' in arch:  # unified BCE (80 classes)
            t = torch.zeros_like(pred_layer[..., 5:])  # targets
            if nb:
                t[b, a, gj, gi, tcls[i]] = 1.0
            lobj += BCE(pred_layer[..., 5:], t)

        elif 'CE' in arch:  # unified CE (1 background + 80 classes)
            t = torch.zeros_like(pred_layer[..., 0], dtype=torch.long)  # targets
            if nb:
                t[b, a, gj, gi] = tcls[i] + 1
            lcls += CE(pred_layer[..., 4:].view(-1, model.nc + 1), t.view(-1))

    lbox *= param['giou']
    lobj *= param['obj']
    lcls *= param['cls']
    loss = lbox + lobj + lcls
    return loss, torch.cat((lbox, lobj, lcls, loss)).detach()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, gamma=0.5, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
