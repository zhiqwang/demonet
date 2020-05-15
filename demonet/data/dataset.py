import os
import numpy as np
import cv2
import torch
import torchvision

from .coco_utils import (
    _coco_remove_images_without_annotations,
    convert_coco_poly_to_mask,
)
from . import transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the dict containing
                object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image_shape = image.shape[1::-1]

        target = dict(
            image_id=img_id,
            image_shape=image_shape,
            annotations=anns,
        )
        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target


def get_coco(
    root,
    image_set,
    transforms,
    mode='instances',
    year=2017,
):
    anno_file_template = "{}_{}{}.json"
    PATHS = {
        "train": ("images", os.path.join(
            "annotations", anno_file_template.format(mode, "train", year),
        )),
        "val": ("images", os.path.join(
            "annotations", anno_file_template.format(mode, "val", year),
        )),
    }

    t = [ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset


def get_dataset(
    name, image_set, transform, data_path,
    mode='instance', year=2017,
):

    dataset = get_coco(
        data_path,
        image_set=image_set,
        transforms=transform,
        mode=mode,
        year=year,
    )

    return dataset


def get_transform(
    is_train=True,
    image_size=300,
    bgr_mean=None,
    bgr_std=None,
):
    transforms = []
    if is_train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.AffineTransform(image_size, random_aug=True))
    else:
        transforms.append(T.ResizeTransform(image_size, image_size))

    transforms.append(T.Normalize(bgr_mean, bgr_std))
    transforms.append(T.ToTensor())
    return Compose(transforms)


def collate_train(batch):
    images = list()
    boxes = list()
    labels = list()

    for image, target in batch:
        images.append(image)
        boxes.append(target['boxes'])
        labels.append(target['labels'])

    images = torch.stack(images, 0)
    targets = {
        'boxes': boxes,
        'labels': labels,
    }

    return images, targets


def collate_eval(batch):
    images = list()
    targets = list()

    for image, target in batch:
        images.append(image)
        targets.append(target)

    images = torch.stack(images, 0)

    return images, targets


class ConvertCocoPolysToMask(object):
    def __call__(self, image, target):
        height, width = image.shape[:2]

        image_id = target["image_id"]
        image_id = np.array([image_id])
        image_shape = target["image_shape"]

        anno = target["annotations"]

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        # BoxMode: convert from XYWH_ABS to XYXY_ABS
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)

        classes = [obj["category_id"] for obj in anno]
        classes = np.array(classes, dtype=np.int64)

        masks = None
        if len(anno[0]["segmentation"]) > 0:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, height, width)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if masks is not None:
            masks = masks[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["image_shape"] = image_shape

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
