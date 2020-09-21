from torch import nn, Tensor
from typing import List

from util.misc import nested_tensor_from_tensor_list


class WrappedDemonet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs: List[Tensor]):
        sample = nested_tensor_from_tensor_list(inputs)
        return self.model(sample)
