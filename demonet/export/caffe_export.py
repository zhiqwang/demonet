import torch

from demonet.modeling.backbone.resnet import resnet18
from demonet.conversion.pytorch.pytorch_parser import PytorchParser
from demonet.conversion.pytorch.engine.hooks import Hook


def parse(model, input, model_name='TransferedPytorchModel'):
    print('>>> Starting transform, this will take a while...')
    hooks = {}
    for name, module in model.named_modules():
        hooks[name] = Hook(module)

    _ = model(input)

    for key, value in hooks.items():
        print('Key: \'{}\'\nInput, length: {}, the first shape: {}\nOutput: {}'.format(
            key, len(hooks[key].input), hooks[key].input[0].shape,
            hooks[key].output.shape,
        ))


def main(args):
    print('>>> Args: {}'.format(args))
    print('>>> Loading pytorch model...')
    model = resnet18(pretrained=True)
    device = torch.device(args.device)
    model = model.to(device)
    model = model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print('>>> Convert pytorch model to CAFFE: {}'.format(args.arch))

    pytorch_parser = PytorchParser(model, dummy_input)
    pytorch_parser.run(args.output_path)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='pytorch caffe converter')
    parser.add_argument('--arch', default='mobilenet_v2',
                        help='model architecture: {} (default: mobilenet_v2)')
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
