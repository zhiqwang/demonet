# De Monet

This repo is about object detection.

Currently work in process, very pleasure for suggestion and cooperation.

## Training (SSD)

```sh
CUDA_VISIBLE_DEVICES=[GPU_ID] python -m demonet.train \
    --dataset voc \
    --train-set trainval \
    --val-set test \
    --dataset-year 2007 2012 \
    --data-path [DATA_PATH] \
    --output-dir [CHECKPOINT_PATH] \
    --epochs [NUM_EPOCHS] \
    --num-classes [NUM_CLASSES] \
    --batch-size 32 \
    --image-size 300 \
    --bgr-mean 104 117 124 \
    --bgr-std 57 57 58 \
    --lr 0.01
```

## Evaluation

```sh
CUDA_VISIBLE_DEVICES=[GPU_ID] python -m demonet.train \
    --data-path [DATA_PATH] \
    --resume [CHECKPOINT_PATH] \
    --num-classes [NUM_CLASSES] \
    --batch-size 32 \
    --image-size 300 \
    --bgr-mean 104 117 124 \
    --bgr-std 57 57 58 \
    --test-only
```
