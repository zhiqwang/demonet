import cv2
import torch

from models import build_model
from util.image import image_transform
from util.overlay import overlay_boxes, overlay_class_names


def main(args):
    print('>>> Args: {}'.format(args))

    device = torch.device('cpu')
    model = build_model(args)
    model.eval()
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    image = image_transform(
        args.image_path,
        input_shape=(300, 300),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
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
    parser.add_argument('--arch', default='ssd_lite_mobilenet_v2',
                        help='model architecture: {} (default: ssd)')
    parser.add_argument('--image-size', default=300, type=int,
                        help='input size of models')
    parser.add_argument('--num-classes', default=21, type=int,
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
