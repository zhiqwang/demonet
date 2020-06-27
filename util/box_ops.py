import torch


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
