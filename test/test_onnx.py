import io

import unittest

import torch
from torchvision.ops._register_onnx_ops import _onnx_opset_version

import onnxruntime

from hubconf import ssd_lite_mobilenet_v2


class ONNXExporterTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(
        self,
        model,
        inputs_list,
        tolerate_small_mismatch=False,
        do_constant_folding=True,
        dynamic_axes=None,
        output_names=None,
        input_names=None,
    ):
        model.eval()

        onnx_io = io.BytesIO()
        # export to onnx with the first input
        torch.onnx.export(
            model,
            inputs_list[0],
            onnx_io,
            do_constant_folding=do_constant_folding,
            opset_version=_onnx_opset_version,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names,
        )
        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                test_ouputs = model(*test_inputs)

            self.ort_validate(onnx_io, test_inputs, test_ouputs, tolerate_small_mismatch)

    def ort_validate(self, onnx_io, inputs, outputs, tolerate_small_mismatch=False):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # compute onnxruntime output prediction
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)
        for i in range(0, len(outputs)):
            try:
                torch.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-04)
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

    def get_image_from_url(self, url, size=None):
        import requests
        from PIL import Image
        from io import BytesIO
        from torchvision import transforms

        data = requests.get(url)
        image = Image.open(BytesIO(data.content)).convert("RGB")
        width, height = image.size
        image_shape = torch.as_tensor([int(height), int(width)])

        if size is None:
            size = (320, 320)
        image = image.resize(size, Image.BILINEAR)

        to_tensor = transforms.ToTensor()
        return (to_tensor(image)[None, :], image_shape[None, :])

    def get_test_images(self):
        image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
        image, image_shape = self.get_image_from_url(url=image_url, size=(320, 320))

        image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
        image2, image_shape2 = self.get_image_from_url(url=image_url2, size=(320, 320))

        images = (image, image_shape)
        test_images = (image2, image_shape2)
        return images, test_images

    def test_ssd_lite_mobilenet_v2(self):
        images, test_images = self.get_test_images()
        x = (torch.rand(1, 3, 320, 320), torch.as_tensor([[320, 320]]))
        model = ssd_lite_mobilenet_v2(
            pretrained=False,
            image_size=320,
            score_thresh=0.5,
            num_classes=21,
        )
        model.eval()
        model(*images)
        # Test exported model on images of different size, or dummy input
        self.run_model(
            model,
            [images, test_images, x],
            input_names=["inputs"],
            output_names=["scores", "labels", "boxes"],
            # dynamic_axes={"inputs": [0, 1, 2, 3], "outputs": [0, 1, 2, 3]},
            tolerate_small_mismatch=True,
        )
        # Test exported model for an image with no detections on other images
        self.run_model(
            model,
            [x, images],
            input_names=["inputs"],
            output_names=["scores", "labels", "boxes"],
            # dynamic_axes={"inputs": [0, 1, 2, 3], "outputs": [0, 1, 2, 3]},
            tolerate_small_mismatch=True,
        )
