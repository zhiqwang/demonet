import unittest

import torch

from models import _utils as det_utils

from models.backbone import MobileNetWithExtraBlocks
from models.prior_box import AnchorGenerator
from models.box_head import MultiBoxLiteHead, SSDBoxHeads

from hubconf import ssd_lite_mobilenet_v2


class ModelTester(unittest.TestCase):

    def test_box_cxcywh_to_xyxy(self):
        t = torch.rand(10, 4)
        r = det_utils.xyxy_to_xywha(det_utils.xywha_to_xyxy(t))
        self.assertLess((t - r).abs().max(), 1e-5)

    def test_mobilenet_with_extra_blocks_script(self):

        model = MobileNetWithExtraBlocks(train_backbone=False)
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
        torch.jit.script(model)

    def _init_test_multibox_head(self):
        hidden_dims = [96, 1280, 512, 256, 256, 64]
        num_anchors = [6, 6, 6, 6, 6, 6]
        num_classes = 21
        box_head = MultiBoxLiteHead(hidden_dims, num_anchors, num_classes)
        return box_head

    def test_multibox_head_script(self):
        model = self._init_test_multibox_head()
        torch.jit.script(model)

    def _init_test_ssd_box_heads(self):
        variances = [0.1, 0.2]
        iou_thresh = 0.5
        negative_positive_ratio = 3
        score_thresh = 0.5
        nms_thresh = 0.45
        post_nms_top_n = 100

        prior_generator = self._init_test_prior_generator()
        multibox_head = self._init_test_multibox_head()

        box_head = SSDBoxHeads(
            prior_generator, multibox_head,
            variances, iou_thresh, negative_positive_ratio,
            score_thresh, nms_thresh, post_nms_top_n,
        )
        return box_head

    def _test_ssd_box_heads_script(self):
        model = self._init_test_ssd_box_heads()
        torch.jit.script(model)

    def _test_ssd_lite_mobilet_v2_script(self):
        model = ssd_lite_mobilenet_v2()
        torch.jit.script(model)


if __name__ == "__main__":
    unittest.main()
