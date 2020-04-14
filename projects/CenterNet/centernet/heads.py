import torch
import torch.nn as nn
import torch.nn.functional as F

from .transform import gather_feature


class CtdetHeads(nn.Module):

    def __init__(self, heads, head_conv, inplanes=256):
        super().__init__()

        self.heads = heads

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(inplanes, head_conv, kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0),
                )
            else:
                fc = nn.Conv2d(inplanes, num_output, kernel_size=1, stride=1, padding=0)
            self.__setattr__(head, fc)

        for head in self.heads:
            final_layer = self.__getattr__(head)
            for _, m in enumerate(final_layer.modules()):
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.weight.shape[0] == self.heads[head]:
                        if 'hm' in head:
                            nn.init.constant_(m.bias, -2.19)
                        else:
                            nn.init.normal_(m.weight, std=0.001)
                            nn.init.constant_(m.bias, 0)

    def forward(self, features, targets=None):
        result = {}
        for head in self.heads:
            result[head] = self.__getattr__(head)(features)

        if targets is None:
            losses = None
        else:
            result['hm'] = torch.sigmoid(result['hm'])
            loss_hm = focal_loss(result['hm'], targets['hm'])

            loss_shape = reg_l1_loss(
                result['shape'],
                targets['mask'],
                targets['index'],
                targets['shape'],
            )

            loss_offset = reg_l1_loss(
                result['offset'],
                targets['mask'],
                targets['index'],
                targets['offset'],
            )

            losses = {
                'hm': loss_hm,
                'shape': loss_shape,
                'offset': loss_offset,
            }

        return result, losses


def focal_loss(pred, gt):
    '''
    Focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation

    Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)
    '''

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(output, mask, index, target):
    pred = gather_feature(output, index, use_transform=True)

    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss
