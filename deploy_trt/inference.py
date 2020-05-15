# Utility functions for performing image inference
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# Modified by Zhiqiang Wang

import os
import time

import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit  # noqa
import numpy as np

from deploy_trt.engine import build_engine, save_engine, load_engine
from deploy_trt.common import allocate_buffers
from deploy_trt.model import ModelData


class TRTInference(object):
    """Manages TensorRT objects for model inference."""
    def __init__(
        self, trt_engine_path, onnx_model_path,
        trt_engine_datatype=trt.DataType.FLOAT,
        calib_dataset=None, batch_size=1,
    ):
        """Initializes TensorRT objects needed for model inference.

        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            onnx_model_path (str): path of .onnx model
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """
        # Suppressed informational messages, and report only warnings and errors
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        self.batch_size = batch_size
        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # TRT engine placeholder
        self.trt_engine = None

        # Display requested engine settings to stdout
        print(">>> TensorRT inference engine settings:")
        print(">>> Inference precision: {}".format(trt_engine_datatype))
        print(">>> Max batch size: {}".format(batch_size))

        # If engine is not cached, we need to build it
        if not os.path.exists(trt_engine_path):
            # This function uses supplied .onnx file
            # alongside with ONNXParser to build TensorRT
            # engine. For more details, check implmentation
            self.trt_engine = build_engine(onnx_model_path, TRT_LOGGER)
            # Save the engine to file
            save_engine(self.trt_engine, trt_engine_path)

        # If we get here, the file with engine exists, so we can load it
        if not self.trt_engine:
            print(">>> Loading cached TensorRT engine from {}".format(trt_engine_path))
            self.trt_engine = load_engine(trt_engine_path, TRT_LOGGER)

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.trt_engine)

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        input_volume = trt.volume(ModelData.INPUT_SHAPE)
        self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))

    def infer(self, image_path):
        """Infers model on given image.

        Args:
            image_path (str): image to run object detection model on
        """

        # Load image into CPU
        img = self._load_img(image_path)

        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, img.ravel())

        # When infering on single image, we measure inference
        # time to output it to the user
        inference_start_time = time.time()

        # Fetch output from the model
        output = self.run_inference()

        # Output inference time
        print(">>> TensorRT inference time: {} ms".format(
            int(round((time.time() - inference_start_time) * 1000)),
        ))

        # And return results
        return output

    def run_inference(self):
        """
        This function is generalized for multiple inputs/outputs.
        inputs and outputs are expected to be lists of HostDeviceMem objects.
        """
        # Transfer input data to the GPU.
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        # Run inference.
        self.context.execute_async(
            batch_size=self.batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        # Transfer predictions back from the GPU.
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image).reshape((
            im_height, im_width, ModelData.get_input_channels(),
        )).astype(np.uint8)

    def _load_imgs(self, image_paths):
        # batch_size = self.trt_engine.max_batch_size
        for idx, image_path in enumerate(image_paths):
            img_np = self._load_img(image_path)
            self.numpy_array[idx] = img_np
        return self.numpy_array

    def _load_img(self, image_path):
        image = Image.open(image_path)
        model_input_width = ModelData.get_input_width()
        model_input_height = ModelData.get_input_height()
        # Note: Bilinear interpolation used by Pillow is a little bit
        # different than the one used by Tensorflow, so if network receives
        # an image that is not 300x300, the network output may differ
        # from the one output by Tensorflow
        image_resized = image.resize(
            size=(model_input_width, model_input_height),
            resample=Image.BILINEAR,
        )
        img_np = self._load_image_into_numpy_array(image_resized)
        # HWC -> CHW
        img_np = img_np.transpose((2, 0, 1))
        # Normalize to [0, 1.0] interval (expected by model)
        img_np = img_np / 255.0
        img_np = img_np.ravel()
        return img_np
