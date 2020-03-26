import os
import torch

from demonet.modeling.backbone.mobilenet import mobilenet_v2
import demonet.export.model_converter as torch2caffe


def main(args):
    args.caffe_model_path = os.path.join(args.output_dir, args.caffe_model_path)
    args.caffe_weight_path = os.path.join(args.output_dir, args.caffe_weight_path)

    print('>>> Args: {}'.format(args))

    print('>>> Loading pytorch model...')
    model = mobilenet_v2()
    device = torch.device(args.device)
    model = model.to(device)
    model = model.eval()

    # load checkpoints
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print('>>> Convert pytorch model to CAFFE: {}'.format(args.arch))

    torch2caffe.trans_net(model, dummy_input, args.arch)  # Import the pytorch model to CAFFE
    torch2caffe.save_prototxt(args.caffe_model_path)
    torch2caffe.save_caffemodel(args.caffe_weight_path)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='pytorch caffe converter')
    parser.add_argument('--arch', default='mobilenet_v2',
                        help='model architecture: {} (default: mobilenet_v2)')
    parser.add_argument('--device', default='cpu',
                        help='device')
    parser.add_argument('--resume', default=None,
                        help='resume from checkpoint')
    parser.add_argument('--output-dir', default='./checkpoints',
                        help='path where to save')
    parser.add_argument('--caffe-model-path', default='deploy.prototxt',
                        help='the path of CAFFE model')
    parser.add_argument('--caffe-weight-path', default='deploy.caffemodel',
                        help='the path of CAFFE weight')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
