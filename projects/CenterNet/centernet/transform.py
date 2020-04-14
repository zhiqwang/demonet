import math
import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor

from torch.jit.annotations import List, Dict, Optional


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


class CtdetTransform(torch.nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, decode_mode='1d'):
        super().__init__()
        self.decode_mode = decode_mode

    def forward(self, images, targets=None):
        return images, targets

    def torch_choice(self, l):
        # type: (List[int])
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(l))).item())
        return l[index]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]])
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = float(torch.min(im_shape))
        max_size = float(torch.max(im_shape))
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int)
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, K=1):
        if self.training:
            return result
        else:
            fmap = torch.sigmoid(result['hm'])
            shape = result['shape']
            offset = result['offset']
            if self.decode_mode == '1d':
                detections = self.decode_1d(fmap, shape, offset, K=K)
            elif self.decode_mode == '2d':
                detections = self.decode_2d(fmap, shape, offset, K=K)
            else:
                raise NotImplementedError(
                    "self.decode_mode: {} should be implemented here!".format(self.decode_mode))

            return detections

    def decode_1d(self, fmap, shape, offset, cat_spec_wh=False, K=100):
        r"""
        decode output feature map to detection results
        Args:
            fmap(Tensor): output feature map
            shape(Tensor): tensor that represents predicted width-height
            offset(Tensor): tensor that represens regression of center points
            cat_spec_wh(bool): whether apply gather on tensor `shape` or not
            K(int): topk value
        """
        batch, channel, height, width = fmap.shape

        # perform nms on heatmaps
        fmap = self.pseudo_nms(fmap)

        scores, index, classes, ys, xs = self.topk_score(fmap, K=K)
        offset = gather_feature(offset, index, use_transform=True)
        offset = offset.reshape(batch, K, 2)
        xs = xs.view(batch, K, 1) + offset[:, :, 0:1]
        ys = ys.view(batch, K, 1) + offset[:, :, 1:2]

        shape = gather_feature(shape, index, use_transform=True)

        if cat_spec_wh:
            shape = shape.view(batch, K, channel, 2)
            clses_ind = classes.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
            shape = shape.gather(2, clses_ind).reshape(batch, K, 2)
        else:
            shape = shape.reshape(batch, K, 1)

        classes = classes.reshape(batch, K, 1).float()
        scores = scores.reshape(batch, K, 1)

        half_h = shape[..., 0:1] / 2
        bboxes = torch.cat([xs, ys - half_h, xs, ys + half_h], dim=2)

        detections = torch.cat([bboxes, scores, classes], dim=2)

        return detections

    def decode_2d(self, fmap, shape, offset, cat_spec_wh=False, K=100):
        r"""
        decode output feature map to detection results
        Args:
            fmap(Tensor): output feature map
            shape(Tensor): tensor that represents predicted width-height
            offset(Tensor): tensor that represens regression of center points
            cat_spec_wh(bool): whether apply gather on tensor `shape` or not
            K(int): topk value
        """
        batch, channel, height, width = fmap.shape

        # perform nms on heatmaps
        fmap = self.pseudo_nms(fmap)

        scores, index, classes, ys, xs = self.topk_score(fmap, K=K)
        offset = gather_feature(offset, index, use_transform=True)
        offset = offset.reshape(batch, K, 2)
        xs = xs.view(batch, K, 1) + offset[:, :, 0:1]
        ys = ys.view(batch, K, 1) + offset[:, :, 1:2]

        shape = gather_feature(shape, index, use_transform=True)

        if cat_spec_wh:
            shape = shape.view(batch, K, channel, 2)
            clses_ind = classes.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
            shape = shape.gather(2, clses_ind).reshape(batch, K, 2)
        else:
            shape = shape.reshape(batch, K, 2)

        classes = classes.reshape(batch, K, 1).float()
        scores = scores.reshape(batch, K, 1)

        half_w, half_h = shape[..., 0:1] / 2, shape[..., 1:2] / 2
        bboxes = torch.cat([xs - half_w, ys - half_h, xs + half_w, ys + half_h], dim=2)

        detections = torch.cat([bboxes, scores, classes], dim=2)

        return detections

    @staticmethod
    def pseudo_nms(fmap, pool_size=3):
        r"""
        apply max pooling to get the same effect of nms
        Args:
            fmap(Tensor): output tensor of previous step
            pool_size(int): size of max-pooling
        """
        pad = (pool_size - 1) // 2
        fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == fmap).float()
        return fmap * keep

    @staticmethod
    def topk_score(scores, K=40):
        """
        get top K point in score map
        """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index / K).int()
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int])
    ratios = [float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size)]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
