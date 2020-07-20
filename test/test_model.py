import unittest

import torch

from models.backbone import MobileNetWithExtraBlocks
from models.prior_box import AnchorGenerator
from models.box_head import MultiBoxLiteHead, PostProcess
from models.generalized_ssd import GeneralizedSSD

from util.misc import nested_tensor_from_tensor_list


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

    def test_prior_generator_script(self):
        model = self._init_test_prior_generator()
        scripted_model = torch.jit.script(model)  # noqa

    def _init_test_multibox_head(self):
        hidden_dims = [96, 1280, 512, 256, 256, 64]
        num_anchors = [6, 6, 6, 6, 6, 6]
        num_classes = 21
        box_head = MultiBoxLiteHead(hidden_dims, num_anchors, num_classes)
        return box_head

    def test_multibox_head_script(self):
        model = self._init_test_multibox_head()
        scripted_model = torch.jit.script(model)  # noqa

    def _init_test_postprocessors(self):
        postprocessors = PostProcess()
        return postprocessors

    def test_postprocessors_script(self):
        model = self._init_test_postprocessors()
        scripted_model = torch.jit.script(model)  # noqa

    def test_ssd_script(self):
        backbone_with_extra_blocks = self._init_test_backbone()
        prior_generator = self._init_test_prior_generator()
        multibox_head = self._init_test_multibox_head()
        post_process = self._init_test_postprocessors()

        model = GeneralizedSSD(backbone_with_extra_blocks, prior_generator, multibox_head, post_process)
        scripted_model = torch.jit.script(model)

        x = nested_tensor_from_tensor_list([torch.rand(3, 320, 320), torch.rand(3, 320, 320)])
        target_sizes = torch.Tensor([[320, 320], [320, 320]])
        out = model(x, target_sizes)
        out_script = scripted_model(x, target_sizes)
        self.assertTrue(out[0]["scores"].equal(out_script[0]["scores"]))
        self.assertTrue(out[0]["labels"].equal(out_script[0]["labels"]))
        self.assertTrue(out[0]["boxes"].equal(out_script[0]["boxes"]))


if __name__ == "__main__":
    unittest.main()
