from itertools import product
import math

import torch
import torch.nn as nn


class PriorBoxGenerator(nn.Module):
    """
    Module that generates priors in XYWHA_REL BoxMode for a set of feature maps and
    image sizes.
    The module support computing priors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.
    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.
    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] priors
    per spatial location for feature map i.
    """
    def __init__(
        self,
        image_size=304,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        feature_maps=[19, 10, 5, 3, 1],
        min_ratio=15,
        max_ratio=90,
        steps=[16, 30, 60, 101, 304],
        clip=True,
    ):
        super().__init__()

        self.aspect_ratios = aspect_ratios
        self.image_size = image_size
        self.feature_maps = feature_maps
        self.num_priors = len(feature_maps)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_sizes, self.max_sizes = self.compute_sizes()
        self.steps = steps
        self.clip = clip

    def compute_sizes(self):
        step = int(math.floor(self.max_ratio - self.min_ratio) / (self.num_priors - 2))
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

    @torch.no_grad()
    def forward(self):
        priors = []
        for k, f in enumerate(self.feature_maps):
            scale = self.image_size / self.steps[k]
            for i, j in product(range(f), repeat=2):
                # unit center x, y
                cx = (j + 0.5) / scale
                cy = (i + 0.5) / scale

                # small sized square box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # big sized square box
                size = math.sqrt(self.min_sizes[k] * self.max_sizes[k])
                h = w = size / self.image_size
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h = w = size / self.image_size
                for ratio in self.aspect_ratios[k]:
                    ratio = math.sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        # back to torch land
        output = torch.as_tensor(priors, dtype=torch.float32)
        if self.clip:
            output.clamp_(min=0, max=1)

        return output
