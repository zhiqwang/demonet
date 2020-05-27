r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3
"""
import datetime
import os
import time

import torch
import torch.utils.data

from .data.dataset import (
    get_dataset,
    get_transform,
    collate_train,
    collate_eval,
)
from .utils.distribute import init_distributed_mode, save_on_master, mkdir

from .engine import train_one_epoch, evaluate

from .modeling.backbone.vgg import build_model


def main(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    dataset = get_dataset(
        args.dataset,
        args.train_set,
        get_transform(is_train=True, bgr_mean=args.bgr_mean, bgr_std=args.bgr_std),
        args.data_path,
        mode=args.dataset_mode,
        years=args.dataset_year,
    )
    dataset_test = get_dataset(
        args.dataset,
        args.val_set,
        get_transform(is_train=False, bgr_mean=args.bgr_mean, bgr_std=args.bgr_std),
        args.data_path,
        mode=args.dataset_mode,
        years=args.dataset_year,
    )

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.batch_size, drop_last=True,
    )
    test_batch_sampler = torch.utils.data.BatchSampler(
        test_sampler, args.batch_size, drop_last=False,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.workers,
        collate_fn=collate_train,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_sampler=test_batch_sampler,
        num_workers=args.workers,
        collate_fn=collate_eval,
    )

    print("Creating model")
    model = build_model(
        size=args.image_size,
        num_classes=args.num_classes,
    )
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
        )
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_steps,
        gamma=args.lr_gamma,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)

        lr_scheduler.step()
        if args.output_dir:
            save_on_master(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                    'epoch': epoch,
                },
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)),
            )

        # evaluate after every epoch
        # evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--data-path', default='./data-bin',
                        help='dataset')
    parser.add_argument('--dataset', default='coco',
                        help='dataset')
    parser.add_argument('--dataset-mode', default='instances',
                        help='dataset mode')
    parser.add_argument('--dataset-year', nargs='+', default=['2017'],
                        help='dataset year')
    parser.add_argument('--train-set', default='train',
                        help='set of train')
    parser.add_argument('--val-set', default='val',
                        help='set of val')
    parser.add_argument('--bgr-mean', type=int, nargs='+',
                        help='mean')
    parser.add_argument('--bgr-std', type=int, nargs='+',
                        help='mean')
    parser.add_argument('--model', default='ssd',
                        help='model')
    parser.add_argument('--device', default='cuda',
                        help='device')
    parser.add_argument('--image-size', default=300, type=int,
                        help='input size of models')
    parser.add_argument('--num-classes', default=21, type=int,
                        help='number classes of datasets')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int,
                        help='print frequency')
    parser.add_argument('--output-dir', default='.',
                        help='path where to save')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=-1, type=int)
    parser.add_argument("--test-only", dest="test_only", action="store_true",
                        help="Only test the model")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained models from the modelzoo")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
