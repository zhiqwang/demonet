import torch
import torchvision

from . import transforms as T


class ConvertVOCtoCOCO(object):

    CLASSES = (
        '__background__', 'aeroplane', 'bicycle',
        'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor',
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

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        ishard = torch.as_tensor(ishard, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        target['ishard'] = ishard

        target['image_id'] = image_id
        target['image_shape'] = image_shape

        return image, target


class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self, img_folder, year, image_set, transforms):
        super().__init__(img_folder, year=year, image_set=image_set)
        self._transforms = transforms
        self.prepare = ConvertVOCtoCOCO()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = {
            'image_id': index,
            'annotations': target['annotation'],
        }
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target


def make_voc_transforms(image_set='train', image_size=300):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train' or image_set == 'trainval':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize([image_size], max_size=1333),
                ])
            ),
            normalize,
        ])
    elif image_set == 'val':
        return T.Compose([
            T.RandomResize([image_size], max_size=500),
            normalize,
        ])
    else:
        raise ValueError(f'unknown {image_set}')


def build(image_set, year, args):

    dataset = VOCDetection(
        img_folder=args.data_path,
        year=year,
        image_set=image_set,
        transforms=make_voc_transforms(image_set=image_set),
    )

    return dataset
