# Utility functions for drawing bounding boxes on PIL images
import numpy as np
import cv2


def draw_bounding_boxes_on_image(
    image,
    boxes,
    color=(255, 0, 0),
    thickness=4,
    display_str_list=(),
):
    """Draws bounding boxes on image.

    Args:
        image (cv2.image): cv2.image object
        boxes (np.array): a 2 dimensional numpy array
            of [N, 4]: (ymin, xmin, ymax, xmax)
            The coordinates are in normalized format between [0, 1]
        color (int, int, int): RGB tuple describing color to draw bounding box
        thickness (int): bounding box line thickness
        display_str_list [str]: list of strings.
            Contains one string for each bounding box.
    Raises:
        ValueError: if boxes is not a [N, 4] array
    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('boxes must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(
            image,
            boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3],
            color, thickness, display_str_list[i],
        )


def draw_bounding_box_on_image(
    image, ymin, xmin, ymax, xmax,
    color=(255, 0, 0), thickness=2,
    display_str='',
    use_normalized_coordinates=True,
):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    The string passed in display_str is displayed above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the string
    is displayed below the bounding box.

    Args:
        image (cv2.image): cv2 image object
        ymin (float): ymin of bounding box
        xmin (float): xmin of bounding box
        ymax (float): ymax of bounding box
        xmax (float): xmax of bounding box
        color (int, int, int): RGB tuple describing color to draw bounding box
        thickness (int): line thickness
        display_str (str): string to display in box
        use_normalized_coordinates (bool): If True, treat coordinates
            ymin, xmin, ymax, xmax as relative to the image. Otherwise treat
            coordinates as absolute
    """
    img_height, img_width = image.shape[:2]

    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            int(xmin * img_width),
            int(xmax * img_width),
            int(ymin * img_height),
            int(ymax * img_height),
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    cv2.rectangle(image, (left, top), (right, bottom), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # If the total height of the display string added to the top of the bounding
    # box exceeds the top of the image, move the string below the bounding box
    # instead of above
    text_width, text_height = cv2.getTextSize(display_str, font, 0.5, 1)[0]
    # Each display_str has a top and bottom margin of 0.05x
    total_display_str_height = (1 + 2 * 0.05) * text_height

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = int(bottom + total_display_str_height)

    margin = np.ceil(0.05 * text_height)

    cv2.rectangle(
        image,
        (left, int(text_bottom - text_height - 2 * margin)),
        (left + text_width, text_bottom),
        color,
        -1,
    )
    cv2.putText(
        image,
        display_str,
        (int(left + margin), int(text_bottom - margin)),
        font,
        0.5,
        (255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
