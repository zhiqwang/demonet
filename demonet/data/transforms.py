# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Feng Wang and Zhiqiang Wang.

import random

import numpy as np
import cv2

import torch

from fvcore.transforms.transform import Transform


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class AffineTransform(Transform):
    """
    Augmentation from CenterNet
    """
    def __init__(self, output_size, boarder=64, random_aug=True):
        """
        Args:
            output_size(int or tuple): a tuple represents (width, height) of image
            boarder(int): boarder size of image
            random_aug(bool): whether apply random augmentation on annos or not
        """
        super().__init__()
        self._set_attributes(locals())
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

    def __call__(self, image, target):
        self.affine = self.get_transform(image)
        image = self.apply_image(image)
        bbox = target["boxes"]
        bbox = self.apply_box(bbox)
        # Filter blank bounding boxes
        keep = (bbox[:, 3] > bbox[:, 1]) & (bbox[:, 2] > bbox[:, 0])
        bbox = bbox[keep]
        classes = target["labels"][keep]

        target["boxes"] = bbox
        target["labels"] = classes

        return image, target

    def get_transform(self, img):
        """
        generate one `AffineTransform` for input image
        """
        img_shape = img.shape[:2]
        center, scale = self.generate_center_and_scale(img_shape)
        src, dst = self.generate_src_and_dst(center, scale, self.output_size)

        return cv2.getAffineTransform(src, dst)

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply AffineTransform for the image(s).
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC.
                The array can be of type uint8 in range [0, 255],
                or floating point in range [0, 1] or [0, 255].
        Returns:
            ndarray: the image(s) after applying affine transform.
        """
        return cv2.warpAffine(img, self.affine, self.output_size, flags=cv2.INTER_LINEAR)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Affine the coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2.
                Each row is (x, y).
        Returns:
            ndarray: the flipped coordinates.
        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        # aug_coords with shape (N, 3), self.affine: (2, 3)
        w, h = self.output_size
        aug_coords = np.column_stack((coords, np.ones(coords.shape[0])))
        coords = np.dot(aug_coords, self.affine.T)
        coords[..., 0] = np.clip(coords[..., 0], 0, w - 1)
        coords[..., 1] = np.clip(coords[..., 1], 0, h - 1)
        return coords

    @staticmethod
    def _get_boarder(boarder, size):
        """
        decide the boarder size of image
        """
        # NOTE This func may be reimplemented in the future
        i = 1
        size //= 2
        while size <= boarder // i:
            i *= 2
        return boarder // i

    def generate_center_and_scale(self, img_shape):
        r"""
        generate center and scale for image randomly
        Args:
            shape(tuple): a tuple represents (height, width) of image
        """
        height, width = img_shape
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = float(max(img_shape))
        if self.random_aug:
            scale = scale * np.random.choice(np.arange(0.9, 1.1, 0.05))
            center[0] = np.random.randint(low=(width // 2 - 5), high=(width // 2 + 5))
            center[1] = np.random.randint(low=(height // 2 - 5), high=(height // 2 + 5))

        return center, scale

    @staticmethod
    def generate_src_and_dst(center, size, output_size):
        r"""
        generate source and destination for affine transform
        """
        if not isinstance(size, np.ndarray) and not isinstance(size, list):
            size = np.array([size, size], dtype=np.float32)
        src = np.zeros((3, 2), dtype=np.float32)
        src_w = size[0]
        src_dir = [0, src_w * -0.5]
        src[0, :] = center
        src[1, :] = src[0, :] + src_dir
        src[2, :] = src[1, :] + (src_dir[1], -src_dir[0])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst_w, dst_h = output_size
        dst_dir = [0, dst_w * -0.5]
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = dst[0, :] + dst_dir
        dst[2, :] = dst[1, :] + (dst_dir[1], -dst_dir[0])

        return src, dst


class RandomHorizontalFlip(Transform):
    """
    Perform horizontal flip.
    """

    def __init__(self, prob: float):
        super().__init__()
        self._set_attributes(locals())

    def __call__(self, image, target):
        if random.random() < self.prob:
            self.width = image.shape[1]
            image = self.apply_image(image)
            bbox = target["boxes"]
            bbox = self.apply_box(bbox)
            target["boxes"] = bbox
        return image, target

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        if img.ndim <= 3:  # HxW, HxWxC
            return np.flip(img, axis=1)
        else:
            return np.flip(img, axis=-2)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.
        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        coords[:, 0] = self.width - coords[:, 0]
        return coords


class ResizeTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, new_h, new_w):
        """
        Args:
            new_h, new_w (int): new image size
        """
        super().__init__()
        self._set_attributes(locals())

    def __call__(self, image, target):
        self.h, self.w = image.shape[:2]  # h, w (int): original image size
        image = self.apply_image(image)
        bbox = target["boxes"]
        bbox = self.apply_box(bbox)
        target["boxes"] = bbox

        return image, target

    def apply_image(self, img):
        img = cv2.resize(img, (self.new_w, self.new_h))
        return img

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords


class Normalize(Transform):
    """
    Normalization
    """
    def __init__(self, mean, std):
        """
        Args:
            mean (list): mean
            std (list): std
        """
        super().__init__()
        self._set_attributes(locals())

    def __call__(self, image, target):

        image = self.apply_image(image)
        bbox = target["boxes"]
        bbox = self.apply_box(bbox)
        target["boxes"] = bbox

        return image, target

    def apply_image(self, img):
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # Therefore it's important to use torch.Tensor.
        img = img.astype("float32")
        if self.mean is not None:
            mean = np.array(self.mean, dtype=np.float32)
            img -= mean[None, None, :]
        if self.std is not None:
            std = np.array(self.std, dtype=np.float32)
            img /= std[None, None, :]
        # Set default scale of image in [0., 1.]
        if self.mean is None and self.std is None:
            img /= 255.
        return img

    def apply_coords(self, coords):
        return coords


class ToTensor:

    def __call__(self, image, target):
        height, width = image.shape[:2]

        image = torch.as_tensor(image.transpose(2, 0, 1))
        # Can use uint8 if it turns out to be slow some day
        # XYXY_ABS BoxMode
        bbox = torch.as_tensor(target["boxes"].astype("float32"))
        # converted XYXY_REL BoxMode
        bbox[:, 0::2] /= width
        bbox[:, 1::2] /= height

        label = torch.as_tensor(target["labels"].astype("int64"))
        img_id = target['image_id']
        img_shape = target['image_shape']

        # Concat the bbox with label
        target = {
            'image_id': img_id,
            'image_shape': img_shape,
            'boxes': bbox,
            'labels': label
        }
        return image, target
