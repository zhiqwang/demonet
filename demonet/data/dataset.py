import os

import torch

from .coco_utils import (
    _coco_remove_images_without_annotations,
    CocoDetection,
    ConvertCocoPolysToMask,
)
from .voc_utils import VOCDetection, ConvertVOCtoCOCO
from . import transforms as T


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
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset


def get_voc(
    root,
    image_set,
    transforms,
):
    t = [ConvertVOCtoCOCO()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    dataset = VOCDetection(
        img_folder=root,
        year='2007',
        image_set=image_set,
        transforms=transforms,
    )

    return dataset


def get_dataset(
    name, image_set, transform, data_path,
    mode='instances', year=2017,
):

    if name == 'coco':
        dataset = get_coco(
            data_path,
            image_set=image_set,
            transforms=transform,
            mode=mode,
            year=year,
        )
    elif name == 'voc':
        dataset = get_voc(
            data_path,
            image_set=image_set,
            transforms=transform,
        )
    else:
        raise NotImplementedError

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
    return T.Compose(transforms)


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
