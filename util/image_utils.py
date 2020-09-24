import numpy as np
import cv2

from IPython import display
from PIL import Image

import torch


def load_image(
    image_name,
    input_shape=(304, 304),
    mean=None,
    std=None,
):
    image = cv2.imread(image_name)
    image = image[:, :, ::-1]  # BGR to RGB
    image = cv2.resize(image, input_shape, interpolation=cv2.INTER_CUBIC)
    image = np.ascontiguousarray(image, dtype=np.float32)  # uint8 to float32
    image /= 255.0  # 0 - 255 to 0.0 - 1.0
    # Normalization
    if mean is not None:
        mean = np.array(mean, dtype=np.float32)  # RGB
        image -= mean
    if std is not None:
        std = np.array(std, dtype=np.float32)  # RGB
        image /= std
    image = image.transpose([2, 0, 1])  # change to C x H x W

    return image


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
        (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
        image.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(Image.fromarray(a))


def select_top_predictions(predictions, threshold):
    idx = torch.where(predictions['scores'] > threshold)[0]
    new_predictions = {}
    for k, v in predictions.items():
        new_predictions[k] = v[idx]
    return new_predictions


def compute_colors_for_labels(labels, palette=None):
    '''
    Simple function that adds fixed colors depending on the class
    Arguments:
        labels (np.ndarray)
        palette (np.ndarray)
    '''
    if isinstance(labels, list):
        labels = np.asarray(labels)
    if palette is None:
        palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).astype('uint8')
    return colors


def overlay_boxes(image, predictions):
    '''
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    '''
    labels = predictions['labels']
    boxes = predictions['boxes']
    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        top_left, bottom_right = tuple(box[:2]), tuple(box[2:])
        image = cv2.rectangle(image, top_left, bottom_right, color, 2)

    return image


def overlay_class_names(image, predictions, categories):
    '''
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    '''
    if isinstance(predictions['scores'], list):
        scores = predictions['scores']
    elif isinstance(predictions['scores'], np.ndarray):
        scores = predictions['scores'].tolist()
    else:
        raise TypeError('scores current only support with type of list or numpy.ndarray')

    labels = predictions['labels']
    colors = compute_colors_for_labels(labels).tolist()
    labels = [categories[i] for i in labels]
    boxes = predictions['boxes']

    template = '{}: {:.2f}'
    for box, score, label, color in zip(boxes, scores, labels, colors):
        x, y = box[:2]
        s = template.format(label, score)
        # Draw black background rectangle
        # cv2.rectangle(image, (x, y), (x + len(s) * 15, y - 20), color, -1)
        color = (0, 0, 0)
        cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)

    return image
