import unittest

import torch
import torchvision

from util.misc import is_main_process
from models.backbone import BackboneBase
from modules.peleenet import peleenet_v1


class TorchScriptTester(unittest.TestCase):

    def test_mobilenet_script(self):
        backbone_base = torchvision.models.mobilenet_v2(pretrained=is_main_process())
        return_layers = {"features": "0"}
        num_channels = 1280

        backbone = BackboneBase(backbone_base, return_layers, True, num_channels)
        torch.jit.script(backbone)  # noqa

    def test_peleenet_script(self):
        backbone_base = peleenet_v1(pretrained=False)
        return_layers = {"features": "0"}
        num_channels = 704

        backbone = BackboneBase(backbone_base, return_layers, True, num_channels)
        torch.jit.script(backbone)  # noqa


if __name__ == "__main__":
    unittest.main()
