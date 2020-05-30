import torch


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
