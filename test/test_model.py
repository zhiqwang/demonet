import unittest

import torch

from demonet.models.backbone import MobileNetWithExtraBlocks
from demonet.models.prior_box import AnchorGenerator
from demonet.models.box_head import MultiBoxLiteHead, PostProcess, SetCriterion
from demonet.models.generalized_ssd import GeneralizedSSD

from demonet.util.misc import nested_tensor_from_tensor_list

from .utils import WrappedDemonet


class ModelTester(unittest.TestCase):

    def _init_test_backbone(self):
        backbone = MobileNetWithExtraBlocks(train_backbone=False)
        return backbone

    def test_mobilenet_with_extra_blocks_script(self):

        model = self._init_test_backbone()
        torch.jit.script(model)

    def _init_test_prior_generator(self):
        image_size = 320
        aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
        min_sizes = [60, 105, 150, 195, 240, 285]
        max_sizes = [105, 150, 195, 240, 285, 330]
        clip = True
        priors_generator = AnchorGenerator(image_size, aspect_ratios, min_sizes, max_sizes, clip)
        return priors_generator

    def _init_test_multibox_head(self):
        hidden_dims = [96, 1280, 512, 256, 256, 64]
        num_anchors = [6, 6, 6, 6, 6, 6]
        num_classes = 21
        box_head = MultiBoxLiteHead(hidden_dims, num_anchors, num_classes)
        return box_head

    def _init_test_postprocessors(self):
        variances = (0.1, 0.2)
        score_thresh = 0.5
        nms_thresh = 0.45
        detections_per_img = 100
        postprocessors = PostProcess(variances, score_thresh, nms_thresh, detections_per_img)
        return postprocessors

    def _init_test_criterion(self):
        variances = (0.1, 0.2)
        iou_thresh = 0.5
        negative_positive_ratio = 3
        criterion = SetCriterion(variances, iou_thresh, negative_positive_ratio)
        return criterion

    def test_prior_generator_script(self):
        model = self._init_test_prior_generator()
        scripted_model = torch.jit.script(model)  # noqa

    def test_multibox_head_script(self):
        model = self._init_test_multibox_head()
        scripted_model = torch.jit.script(model)  # noqa

    def test_postprocessors_script(self):
        model = self._init_test_postprocessors()
        scripted_model = torch.jit.script(model)  # noqa

    def _test_criterion_script(self):
        model = self._init_test_criterion()
        scripted_model = torch.jit.script(model)  # noqa

    def test_ssd_script(self):
        backbone = self._init_test_backbone()
        prior_generator = self._init_test_prior_generator()
        multibox_head = self._init_test_multibox_head()
        post_process = self._init_test_postprocessors()

        model = GeneralizedSSD(backbone, prior_generator, multibox_head, post_process)
        scripted_model = torch.jit.script(model)

        model.eval()
        scripted_model.eval()

        x = nested_tensor_from_tensor_list([torch.rand(3, 320, 320), torch.rand(3, 320, 320)])

        out = model(x)
        out_script = scripted_model(x)[1]
        self.assertTrue(out[0]["scores"].equal(out_script[0]["scores"]))
        self.assertTrue(out[0]["labels"].equal(out_script[0]["labels"]))
        self.assertTrue(out[0]["boxes"].equal(out_script[0]["boxes"]))

    def test_wrapped_ssd_script(self):
        backbone = self._init_test_backbone()
        prior_generator = self._init_test_prior_generator()
        multibox_head = self._init_test_multibox_head()
        post_process = self._init_test_postprocessors()

        model = GeneralizedSSD(backbone, prior_generator, multibox_head, post_process)
        wrapped_model = WrappedDemonet(model)
        scripted_model = torch.jit.script(wrapped_model)

        wrapped_model.eval()
        scripted_model.eval()

        x = [torch.rand(3, 320, 320), torch.rand(3, 320, 320)]

        out = wrapped_model(x)
        out_script = scripted_model(x)[1]
        self.assertTrue(out[0]["scores"].equal(out_script[0]["scores"]))
        self.assertTrue(out[0]["labels"].equal(out_script[0]["labels"]))
        self.assertTrue(out[0]["boxes"].equal(out_script[0]["boxes"]))
