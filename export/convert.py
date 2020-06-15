import torch

from converter.pytorch.pytorch_parser import PytorchParser
from brocolli.modeling.backbone.peleenet import build_model


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
    model.load_state_dict(checkpoint['model'], strict=False)
    # dummy_input = torch.ones([1, 3, 304, 304])

    pytorch_parser = PytorchParser(model, [3, 304, 304])
    pytorch_parser.run(args.output_path)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='pytorch ssd converter')
    parser.add_argument('--arch', default='ssd',
                        help='model architecture: {} (default: ssd)')
    parser.add_argument('--image-size', default=304, type=int,
                        help='input size of models')
    parser.add_argument('--num-classes', default=4, type=int,
                        help='number classes of datasets')
    parser.add_argument('--config', default='./ssd/config/Pelee_COCO.py')
    parser.add_argument('--device', default='cpu',
                        help='device')
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')
    parser.add_argument('--output-path', default='./checkpoints',
                        help='path where to save')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
