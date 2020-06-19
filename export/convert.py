import torch

from converter.torch_tools.pytorch_parser import PytorchParser
from hubconf import ssd_lite_mobilenet_v2


def main(args):
    print('>>> Args: {}'.format(args))

    device = torch.device('cpu')

    model = ssd_lite_mobilenet_v2(
        pretrained=True,
        num_classes=args.num_classes,
        image_size=300,
    )
    model.eval()
    model.to(device)

    # dummy_input = torch.ones([1, 3, 300, 300])

    pytorch_parser = PytorchParser(model, [3, 300, 300])
    pytorch_parser.run(args.output_path)


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
    parser.add_argument('--output-path', default='./checkpoints',
                        help='path where to save')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
