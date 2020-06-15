# De Monet

This repo is about object detection.

Currently work in process, very pleasure for suggestion and cooperation.

## Training (SSD)

```sh
CUDA_VISIBLE_DEVICES=[GPU_ID] python -m train \
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
    --lr 0.01
```

## Evaluation

### Evaluation on coco dataset

```sh
CUDA_VISIBLE_DEVICES=[GPU_ID] python -m train \
    --arch ssd_lite_mobilenet_v2 \
    --image-size 300 \
    --dataset-file coco \
    --dataaset-mode pascal \
    --val-set test \
    --dataset-year 2007 \
    --data-path [DATA_PATH] \
    --resume [CHECKPOINT_PATH] \
    --num-classes [NUM_CLASSES] \
    --batch-size 32 \
    --test-only
```

### Evaluation on voc dataset

```sh
CUDA_VISIBLE_DEVICES=[GPU_ID] python -m eval_voc \
    --arch ssd_lite_mobilenet_v2 \
    --image-size 300 \
    --dataset-file voc \
    --val-set test \
    --dataset-year 2007 \
    --data-path [VOC_DEVKIT_ROOT_PATH] \
    --num-classes [NUM_CLASSES] \
    --batch-size 32 \
    --resume [CHECKPOINT_PATH] \
    --output-dir [OUTPUT_DIR]
```
