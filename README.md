# 👿 De Monet - All of Object Detection

[![Test](https://github.com/zhiqwang/demonet/workflows/Test/badge.svg)](https://github.com/zhiqwang/demonet/actions?query=workflow%3ATest)

PyTorch training code and models reimplentation for object detection as described in [Liu et al. (2015), SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325). *Currently work in process, very pleasure for suggestion and cooperation.*

![Example of SSD Lite with mobilenet v2 backbone](.github/demo.png)

## 🆕 What's New and Development Plans

- [x] Support exporting to `TorchScript` model. *Jul. 22, 2020.*
- [x] Support exporting to `onnx`, and doing inference using `onnxruntime`. *Jul. 25, 2020.*
- [x] Support doing inference using `libtorch` cpp interface. *Sep. 18, 2020.*
- [ ] Add more fetures ...

## 🛠 Usage

There are no extra compiled components in DEMONET and package dependencies are minimal, so the code is very simple to use. We provide instructions how to install dependencies via conda. First, clone the repository locally:

```bash
git clone https://github.com/zhiqwang/demonet.git
```

Then, install PyTorch 1.6+ and torchvision 0.7+:

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Install pycocotools (for evaluation on COCO) and scipy (for training):

```bash
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

That's it, should be good to train and evaluate detection models.

## 🧗 Data Preparation

Support trainint with COCO and PASCAL VOC format (chosen with the parameter `--dataset-file [coco/voc]`). With COCO format we expect the directory structure to be the following:

```bash
.
└── path/to/data-path/
    ├── annotations  # annotation json files
    └── images       # root path of images
```

When you are using PASCAL VOC format, we expect the directory structure to be the following:

```bash
.
└── path/to/data-path/
    └── VOCdevkit
        ├── VOC2007
        └── VOC2012
```

## 🦄 Training and Evaluation Snippets

```
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --data-path 'data-bin/mscoco/coco2017/' --dataset coco --model ssdlite320_mobilenet_v3_large --pretrained --test-only
```

## 🎓 Acknowledgement

- This repo borrows the architecture design and part of the code from [DETR](https://github.com/facebookresearch/detr) and [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models/detection).
- The implementation of `ssd_lite_mobilenet_v2` borrow the code from [qfgaohao's pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) and [lufficc's SSD](http://github.com/lufficc/SSD/).
