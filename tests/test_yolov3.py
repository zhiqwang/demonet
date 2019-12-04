import unittest
import torch

from demonet.modeling.yolov3 import Darknet


class YOLOV3Test(unittest.TestCase):

    def test_yolov3(self):

        device = torch.device('cuda:2')

        model = Darknet('./demonet/modeling/config/yolov3.cfg').to(device)

        model = model.to(device)
        input_shape = (32, 3, 416, 416)
        x = torch.randn(input_shape).to(device)
        y = model(x)

        self.assertEqual(len(y), 3)

        self.assertEqual(tuple(y[0].shape), (32, 3, 13, 13, 85))
        self.assertEqual(tuple(y[1].shape), (32, 3, 26, 26, 85))
        self.assertEqual(tuple(y[2].shape), (32, 3, 52, 52, 85))


if __name__ == "__main__":
    unittest.main()
