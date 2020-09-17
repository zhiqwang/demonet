import os

import torch

from hubconf import ssd_lite_mobilenet_v2

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.dirname(os.path.dirname(HERE))

model = ssd_lite_mobilenet_v2(pretrained=False)
model.eval()

traced_model = torch.jit.script(model)
traced_model.save("./test/tracing/ssd_lite_mobilenet_v2.pt")
