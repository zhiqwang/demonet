import xml.etree.ElementTree as ET

import numpy as np
import cv2

import torchvision


class ConvertVOCtoCOCO(object):

    CLASSES = (
        "__background__", "aeroplane", "bicycle",
        "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor",
    )

    def __call__(self, image, target):
        # return image, target
        anno = target['annotations']
        image_id = target['image_id']
        height, width = anno['size']['height'], anno['size']['width']
        image_shape = [width, height]
        boxes = []
        classes = []
        ishard = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj['name']))
            ishard.append(int(obj['difficult']))

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        classes = np.asarray(classes, dtype=np.int64)
        ishard = np.asarray(ishard, dtype=np.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["ishard"] = ishard

        target['image_id'] = image_id
        target["image_shape"] = image_shape

        return image, target


class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self, img_folder, year, image_set, transforms):
        super().__init__(img_folder, year=year, image_set=image_set)
        self._transforms = transforms

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        target = dict(
            image_id=index,
            annotations=target['annotation'],
        )
        if self._transforms is not None:
            image, target = self._transforms(image, target)

        return image, target
