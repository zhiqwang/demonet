import math

import torch
from torch import nn, Tensor
from torch.jit.annotations import List, Optional, Dict


class AnchorGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    """
    Module that generates priors in XYWHA_REL BoxMode for a set of feature maps and
    image sizes.

    The module support computing priors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes (min_sizes, max_sizes) and aspect_ratios should have the same number of
    elements, and it should correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of (2 * len(aspect_ratios[i]) + 2) priors
    per spatial location for feature map i.
    """
    def __init__(
        self,
        image_size=300,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        min_ratio=20,
        max_ratio=80,
        steps=[16, 32, 64, 100, 150, 300],
        clip=True,
        min_sizes=[60, 105, 150, 195, 240, 285],
        max_sizes=[105, 150, 195, 240, 285, 330],
    ):
        super().__init__()
        self.image_size = image_size
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        if min_sizes is not None:
            self.min_sizes, self.max_sizes = min_sizes, max_sizes
        else:
            self.min_sizes, self.max_sizes = self.compute_sizes()
        assert len(self.min_sizes) == len(self.max_sizes)

        self.steps = tuple(steps)
        self.sizes = tuple((s,) for s in self.min_sizes)
        assert len(self.sizes) == len(self.steps)

        self.aspect_ratios, self.scale_ratios = self.compute_ratios(aspect_ratios)
        assert len(self.sizes) == len(self.aspect_ratios)

        self.clip = clip
        self.cell_anchors = None
        self._cache = {}

    def compute_sizes(self):
        step = int(math.floor(self.max_ratio - self.min_ratio) / (len(self.aspect_ratios) - 2))
        min_sizes = list()
        max_sizes = list()
        for ratio in range(self.min_ratio, self.max_ratio + 1, step):
            min_sizes.append(self.image_size * ratio / 100)
            max_sizes.append(self.image_size * (ratio + step) / 100)

        if self.min_ratio == 20:
            min_sizes = [self.image_size * 10 / 100.] + min_sizes
            max_sizes = [self.image_size * 20 / 100.] + max_sizes
        else:
            min_sizes = [self.image_size * 7 / 100.] + min_sizes
            max_sizes = [self.image_size * 15 / 100.] + max_sizes

        return min_sizes, max_sizes

    def compute_ratios(self, aspect_ratios):
        ratios = []
        scale_ratios = []
        for k, aspect_ratio in enumerate(aspect_ratios):
            ratio = [1.0, 1.0]
            extra = self.max_sizes[k] / self.min_sizes[k]
            scale_ratio = [1.0] * (2 + 2 * len(aspect_ratio))
            scale_ratio[1] = extra  # scale for extra prior
            for r in aspect_ratio:
                ratio.append(1 / r)
                ratio.append(r)

            ratios.append(ratio)
            scale_ratios.append(scale_ratio)

        assert len(ratios) == len(scale_ratios)
        return tuple(tuple(r) for r in ratios), tuple(tuple(s) for s in scale_ratios)

    # TODO: https://github.com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor
    #   with those values in XYWHA BoxMode.
    # (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(self, scales, aspect_ratios, scale_ratios, dtype=torch.float32, device="cpu"):
        # type: (List[int], List[float], List[float], int, Device) -> Tensor  # noqa: F821
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        scale_ratios = torch.sqrt(torch.as_tensor(scale_ratios, dtype=dtype, device=device))
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :] * scale_ratios[:, None]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :] * scale_ratios[:, None]).view(-1)

        zeros = torch.zeros_like(ws)
        base_anchors = torch.stack([zeros, zeros, ws, hs], dim=1)
        return base_anchors

    def set_cell_anchors(self, dtype, device):
        # type: (int, Device) -> None  # noqa: F821
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        cell_anchors = [
            self.generate_anchors(
                sizes, aspect_ratios, scale_ratios, dtype, device,
            )
            for sizes, aspect_ratios, scale_ratios in zip(
                self.sizes, self.aspect_ratios, self.scale_ratios
            )
        ]
        self.cell_anchors = cell_anchors

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            shifts_x = (
                torch.arange(0, grid_width, dtype=torch.float32, device=device) + 0.5
            ) * stride_width
            shifts_y = (
                torch.arange(0, grid_height, dtype=torch.float32, device=device) + 0.5
            ) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            zeros = torch.zeros_like(shift_x)
            shifts = torch.stack((shift_x, shift_y, zeros, zeros), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, feature_maps):
        # type: (List[Tensor]) -> Tensor
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = list((s, s) for s in self.steps)
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors_in_image = torch.jit.annotate(List[torch.Tensor], [])
        for anchors_per_feature_map in anchors_over_all_feature_maps:
            anchors_in_image.append(anchors_per_feature_map)
        anchors = torch.cat(anchors_in_image) / self.image_size
        if self.clip:
            anchors.clamp_(min=0.0, max=1.0)
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors
