import torch
from torchvision.ops._register_onnx_ops import _onnx_opset_version

from hubconf import ssd_lite_mobilenet_v2


def main(args):
    print('>>> Args: {}'.format(args))

    device = torch.device('cpu')

    model = ssd_lite_mobilenet_v2(
        pretrained=True,
        num_classes=args.num_classes,
        image_size=args.image_size,
    )
    model.eval()
    model.to(device)

    image = torch.ones([1, 3, args.image_size, args.image_size]).to(device)
    img_shape = torch.as_tensor([[args.image_size, args.image_size]]).to(device)
    sample_input = (image, img_shape)

    torch.onnx.export(
        model,
        sample_input,
        args.output_path,
        do_constant_folding=True,
        opset_version=_onnx_opset_version,
        input_names=['inputs'],
        output_names=['labels', 'scores', 'boxes'],
    )


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='pytorch ssd converter')
    parser.add_argument('--arch', default='ssd_lite_mobilenet_v2',
                        help='model architecture: {} (default: ssd)')
    parser.add_argument('--image-size', default=320, type=int,
                        help='input size of models')
    parser.add_argument('--num-classes', default=21, type=int,
                        help='number classes of datasets')
    parser.add_argument('--device', default='cpu',
                        help='device')
    parser.add_argument('--output-path', default='./checkpoints/model.onnx',
                        help='path where to save')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
