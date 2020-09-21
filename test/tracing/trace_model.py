import torch

from hubconf import ssd_lite_mobilenet_v2
from test.utils import WrappedDemonet


if __name__ == "__main__":

    model = ssd_lite_mobilenet_v2(pretrained=False)
    wrapped_model = WrappedDemonet(model)
    wrapped_model.eval()

    traced_model = torch.jit.script(wrapped_model)
    traced_model.save("./test/tracing/ssd_lite_mobilenet_v2.pt")
