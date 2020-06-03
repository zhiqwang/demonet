# De Monet

This repo is about object detection.

Currently work in process, very pleasure for suggestion and cooperation.

## Training (SSD)

```sh
CUDA_VISIBLE_DEVICES=[GPU_ID] python -m demonet.train \
    --arch ssd_lite_mobilenet_v2 \
    --image-size 300 \
    --dataset-file voc \
    --train-set trainval \
    --val-set test \
    --dataset-year 2007 2012 \
    --data-path [DATA_PATH] \
    --output-dir [CHECKPOINT_PATH] \
    --epochs [NUM_EPOCHS] \
    --num-classes [NUM_CLASSES] \
    --batch-size 32 \
    --image-size 304 \
    --lr 0.01
```

## Evaluation

```sh
CUDA_VISIBLE_DEVICES=[GPU_ID] python -m demonet.train \
    --arch ssd_lite_mobilenet_v2 \
    --image-size 300 \
    --dataset-file voc \
    --val-set test \
    --dataset-year 2007 2012 \
    --data-path [DATA_PATH] \
    --resume [CHECKPOINT_PATH] \
    --num-classes [NUM_CLASSES] \
    --batch-size 32 \
    --image-size 304 \
    --test-only
```
