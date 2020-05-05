import cv2
import torch

from .modeling.backbone.vgg import build_model
from .config.CC import Config
from .utils.image import image_transform
from .utils.overlay import overlay_boxes, overlay_class_names


def main(args):
    print('>>> Args: {}'.format(args))
    cfg = Config.fromfile(args.config)
    device = torch.device('cpu')
    model = build_model(
        'test',
        size=cfg.model.input_size,
        num_classes=cfg.model.num_classes,
        model_config=cfg.model,
    )
    model.eval()
    model.to(device)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint)

    image = image_transform(
        args.image_path,
        input_shape=(300, 300),
        mean=[103.94, 116.78, 123.68],  # BGR
    )

    image = image[None, :]
    image = torch.from_numpy(image).to(device)

    with torch.no_grad():
        detections = model(image)

    if args.overlay_result:
        image = cv2.imread(args.image_path)
        image_shape = torch.Tensor(image.shape[1::-1]).repeat(2)
        categories = None
        predictions = parse_output(detections, image_shape)
        image = overlay_boxes(image, predictions)
        image = overlay_class_names(image, predictions, categories)


def parse_output(detections, image_shape, threshold=0.6):
    # Parse the outputs
    predictions = {}
    # detection format: label, score, xmin, ymin, xmax, ymax
    det_label = detections[0, :, 0]
    det_conf = detections[0, :, 1]
    det_boxes = detections[0, :, 2:]

    # Get detections with confidence higher than threshold.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= threshold]

    top_label = det_label[top_indices]
    top_conf = det_conf[top_indices]
    top_boxes = det_boxes[top_indices, :]
    top_boxes = top_boxes * image_shape

    predictions = {}
    predictions['labels'] = top_label.cpu().numpy().astype(int)
    predictions['scores'] = top_conf.cpu().numpy()
    predictions['boxes'] = []
    for box in top_boxes.cpu().numpy().astype(int):
        predictions['boxes'].append(box)

    return predictions


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='pytorch ssd converter')
    parser.add_argument('--arch', default='ssd',
                        help='model architecture: {} (default: ssd)')
    parser.add_argument('--config', default='./ssd/config/Pelee_COCO.py')
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
