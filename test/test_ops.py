import unittest
import torch

from util import box_ops


class OpsTester(unittest.TestCase):

    def test_box_cxcywh_to_xyxy(self):
        t = torch.rand(10, 4)
        r = box_ops.xyxy_to_xywha(box_ops.xywha_to_xyxy(t))
        self.assertLess((t - r).abs().max(), 1e-5)


if __name__ == "__main__":
    unittest.main()
