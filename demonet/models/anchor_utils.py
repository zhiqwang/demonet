import math

import torch
from torch import nn, Tensor
from torchvision.models.detection.image_list import ImageList

from typing import List, Optional, Dict, Tuple


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

    Arguments:
        image_size (int): resized image size.
        aspect_ratios (List[List[int]]): optional aspect ratios of the boxes. can be multiple
        min_sizes (List[int]): minimum box size in pixels. can be multiple. required!.
        max_sizes (List[int]): maximum box size in pixels. can be ignored or same as the of min_size.
        clip (bool): whether clip prior boxes.
    """
    def __init__(
        self,
        image_size=320,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        min_sizes=[60, 105, 150, 195, 240, 285],
        max_sizes=[105, 150, 195, 240, 285, 330],
        clip=True,
    ):
        super().__init__()
        self.image_size = image_size

        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        assert len(self.min_sizes) == len(self.max_sizes)

        self.sizes = tuple((s,) for s in self.min_sizes)
        self.aspect_ratios, self.scale_ratios = self.compute_ratios(aspect_ratios)
        assert len(self.sizes) == len(self.aspect_ratios)

        self.clip = clip
        self.cell_anchors = None
        self._cache = {}

    def compute_ratios(self, aspect_ratios):
        # type: (List(float)) -> Tuple(List(float), List(float))
        ratios = []
        scale_ratios = []
        for k, aspect_ratio in enumerate(aspect_ratios):
            aspect_ratio = [float(i) for i in aspect_ratio]
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

    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        scale_ratios: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cpu",
    ) -> Tensor:
        """
        TODO: https://github.com/pytorch/pytorch/issues/26792
        For every (aspect_ratios, scales) combination, output a zero-centered anchor
          with those values in XYWHA BoxMode.
        (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
        This method assumes aspect ratio = height / width for an anchor.
        """
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

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
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
        strides = [[torch.tensor(self.image_size // g[0], dtype=torch.int64, device=device),
                    torch.tensor(self.image_size // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
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


class DefaultBoxGenerator(nn.Module):
    """
    This module generates the default boxes of SSD for a set of feature maps and image sizes.

    Args:
        aspect_ratios (List[List[int]]): A list with all the aspect ratios used in each feature map.
        min_ratio (float): The minimum scale :math:`\text{s}_{\text{min}}` of the default boxes used in the estimation
            of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
        max_ratio (float): The maximum scale :math:`\text{s}_{\text{max}}`  of the default boxes used in the estimation
            of the scales of each feature map. It is used only if the ``scales`` parameter is not provided.
        scales (List[float]], optional): The scales of the default boxes. If not provided it will be estimated using
            the ``min_ratio`` and ``max_ratio`` parameters.
        steps (List[int]], optional): It's a hyper-parameter that affects the tiling of defalt boxes. If not provided
            it will be estimated from the data.
        clip (bool): Whether the standardized values of default boxes should be clipped between 0 and 1. The clipping
            is applied while the boxes are encoded in format ``(cx, cy, w, h)``.
    """

    def __init__(self, aspect_ratios: List[List[int]], min_ratio: float = 0.15, max_ratio: float = 0.9,
                 scales: Optional[List[float]] = None, steps: Optional[List[int]] = None, clip: bool = True):
        super().__init__()
        if steps is not None:
            assert len(aspect_ratios) == len(steps)
        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.clip = clip
        num_outputs = len(aspect_ratios)

        # Estimation of default boxes scales
        if scales is None:
            if num_outputs > 1:
                range_ratio = max_ratio - min_ratio
                self.scales = [min_ratio + range_ratio * k / (num_outputs - 1.0) for k in range(num_outputs)]
                self.scales.append(1.0)
            else:
                self.scales = [min_ratio, max_ratio]
        else:
            self.scales = scales

        self._wh_pairs = self._generate_wh_pairs(num_outputs)

    def _generate_wh_pairs(self, num_outputs: int, dtype: torch.dtype = torch.float32,
                           device: torch.device = torch.device("cpu")) -> List[Tensor]:
        _wh_pairs: List[Tensor] = []
        for k in range(num_outputs):
            # Adding the 2 default width-height pairs for aspect ratio 1 and scale s'k
            s_k = self.scales[k]
            s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
            wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            # Adding 2 pairs for each aspect ratio of the feature map k
            for ar in self.aspect_ratios[k]:
                sq_ar = math.sqrt(ar)
                w = self.scales[k] * sq_ar
                h = self.scales[k] / sq_ar
                wh_pairs.extend([[w, h], [h, w]])

            _wh_pairs.append(torch.as_tensor(wh_pairs, dtype=dtype, device=device))
        return _wh_pairs

    def num_anchors_per_location(self):
        # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * ratios of feaure map.
        return [2 + 2 * len(r) for r in self.aspect_ratios]

    # Default Boxes calculation based on page 6 of SSD paper
    def _grid_default_boxes(self, grid_sizes: List[List[int]], image_size: List[int],
                            dtype: torch.dtype = torch.float32) -> Tensor:
        default_boxes = []
        for k, f_k in enumerate(grid_sizes):
            # Now add the default boxes for each width-height pair
            if self.steps is not None:
                x_f_k, y_f_k = [img_shape / self.steps[k] for img_shape in image_size]
            else:
                y_f_k, x_f_k = f_k

            shifts_x = ((torch.arange(0, f_k[1]) + 0.5) / x_f_k).to(dtype=dtype)
            shifts_y = ((torch.arange(0, f_k[0]) + 0.5) / y_f_k).to(dtype=dtype)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack((shift_x, shift_y) * len(self._wh_pairs[k]), dim=-1).reshape(-1, 2)
            # Clipping the default boxes while the boxes are encoded in format (cx, cy, w, h)
            _wh_pair = self._wh_pairs[k].clamp(min=0, max=1) if self.clip else self._wh_pairs[k]
            wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1)

            default_box = torch.cat((shifts, wh_pairs), dim=1)

            default_boxes.append(default_box)

        return torch.cat(default_boxes, dim=0)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'aspect_ratios={aspect_ratios}'
        s += ', clip={clip}'
        s += ', scales={scales}'
        s += ', steps={steps}'
        s += ')'
        return s.format(**self.__dict__)

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        default_boxes = self._grid_default_boxes(grid_sizes, image_size, dtype=dtype)
        default_boxes = default_boxes.to(device)

        dboxes = []
        for _ in image_list.image_sizes:
            dboxes_in_image = default_boxes
            dboxes_in_image = torch.cat([dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:],
                                         dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:]], -1)
            dboxes_in_image[:, 0::2] *= image_size[1]
            dboxes_in_image[:, 1::2] *= image_size[0]
            dboxes.append(dboxes_in_image)
        return dboxes
