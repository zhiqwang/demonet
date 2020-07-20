import time
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

from models import build_model
from util.misc import MetricLogger, collate_fn

from datasets import build_dataset
from datasets.voc_eval import _write_voc_results_file, _do_python_eval


def main(args):
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    dataset = build_dataset(args.val_set, args.dataset_year, args)

    print("Creating data loaders")
    sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = DataLoader(
        dataset,
        args.batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    print("Creating model")
    model = build_model(args)
    model.to(device)

    # load model weights
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint)

    output_dir = Path(args.output_dir)
    # evaluation
    evaluate(model, data_loader, device, output_dir)


@torch.no_grad()
def evaluate(model, data_loader, device, output_dir):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    cls_names = data_loader.dataset.prepare.CLASSES

    all_boxes = [[] for i in range(len(cls_names))]
    image_index = []

    for samples, targets in metric_logger.log_every(data_loader, 20, header):
        samples = samples.to(device)

        model_time = time.time()
        target_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(device)
        results = model(samples, target_sizes)

        model_time = time.time() - model_time

        for target, result in zip(targets, results):
            image_index.append(''.join([chr(i) for i in target['filename'].tolist()]))

            # Convert the result of models to numpy
            boxes = result['boxes'].tolist()
            labels = result['labels'].tolist()
            scores = result['scores'].tolist()

            image_boxes = [[] for i in range(len(cls_names))]
            for i, box in enumerate(boxes):
                cls_dets = np.hstack((box, scores[i]))
                image_boxes[labels[i]].append(cls_dets)

            for i in range(len(cls_names)):
                if image_boxes[i] != []:
                    all_boxes[i].append(image_boxes[i])
                else:
                    all_boxes[i].append([])

        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    _write_voc_results_file(all_boxes, image_index, cls_names, output_dir)
    _do_python_eval(data_loader, output_dir, use_07=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--arch', default='ssd_lite_mobilenet_v2',
                        help='model architecture')
    parser.add_argument('--data-path', default='./data-bin',
                        help='dataset')
    parser.add_argument('--dataset-file', default='voc',
                        help='dataset')
    parser.add_argument('--dataset-year', default='2007', nargs='+',
                        help='dataset year')
    parser.add_argument('--train-set', default='train',
                        help='set of train')
    parser.add_argument('--val-set', default='val',
                        help='set of val')
    parser.add_argument("--masks", action="store_true",
                        help="semantic segmentation")
    parser.add_argument('--device', default='cuda',
                        help='device')
    parser.add_argument('--score-thresh', default=0.01, type=float,
                        help='inference score threshold')
    parser.add_argument("--onnx-export", action="store_true",
                        help="Whether to export the model to onnx")
    parser.add_argument('--image-size', default=300, type=int,
                        help='input size of models')
    parser.add_argument('--num-classes', default=21, type=int,
                        help='number classes of datasets')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--num-workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr-backbone', default=-1, type=float)
    parser.add_argument('--print-freq', default=20, type=int,
                        help='print frequency')
    parser.add_argument('--output-dir', default='.',
                        help='path where to save')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained models from the modelzoo")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
