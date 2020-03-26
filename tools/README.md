# Export to other deep learning frameworks

## Export to caffe
```
python -m demonet.export.caffe_export \
    --output-dir ./checkpoints/imagenet/ \
    --caffe-model-path mobilenet_v2.prototxt \
    --caffe-weight-path mobilenet_v2.caffemodel
```

## Inference with caffe
```
python -m demonet.export.inference_caffe \
    --output-dir ./checkpoints/imagenet/ \
    --caffe-model-path mobilenet_v2.prototxt \
    --caffe-weight-path mobilenet_v2.caffemodel \
    --image-path [test.jpg]
```
