import torch
from centernet import centernet_resnet18


def test_load_model():
    heads = {'hm': 21, 'shape': 1, 'offset': 2}
    model = centernet_resnet18(heads)
    model.eval()
    dummy_input = torch.randn(4, 3, 256, 256)
    output = model(dummy_input)
    print(output.shape)


if __name__ == "__main__":
    test_load_model()
