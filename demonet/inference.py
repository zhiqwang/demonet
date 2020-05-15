import cv2
import torch

from .modeling.backbone.vgg import build_model
from .utils.image import image_transform
from .utils.overlay import overlay_boxes, overlay_class_names


def main(args):
    print('>>> Args: {}'.format(args))

    device = torch.device('cpu')
    model = build_model(
        size=args.image_size,
        num_classes=args.num_classes,
    )
    model.eval()
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint)

    image = image_transform(
        args.image_path,
        input_shape=(300, 300),
        mean=[104, 117, 124],  # BGR
    )

    image = image[None, :]
    image = torch.from_numpy(image).to(device)

    with torch.no_grad():
        detections = model(image)

    if args.overlay_result:
        image = cv2.imread(args.image_path)
        categories = None
        image = overlay_boxes(image, detections)
        image = overlay_class_names(image, detections, categories)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='pytorch ssd converter')
    parser.add_argument('--arch', default='ssd',
                        help='model architecture: {} (default: ssd)')
    parser.add_argument('--image-size', default=304, type=int,
                        help='input size of models')
    parser.add_argument('--num-classes', default=4, type=int,
                        help='number classes of datasets')
    parser.add_argument('--device', default='cpu',
                        help='device')
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')
    parser.add_argument('--image-path', default='demo.bmp',
                        help='Test image path')
    parser.add_argument('--output-path', default='./checkpoints',
                        help='path where to save')
    parser.add_argument('--overlay-result', action='store_true',
                        help='overlay result')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
