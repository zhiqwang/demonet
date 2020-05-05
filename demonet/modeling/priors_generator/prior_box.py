from itertools import product
import math

import torch

from ..box_heads.box_utils import xywha_to_xyxy
from .anchor_utils import config_parse, assign_priors, encode


class TargetsPriorsAssignor:
    def __init__(self, variances, iou_threshold):
        self.variances = variances
        self.iou_threshold = iou_threshold

    def __call__(self, priors_xywha, targets):
        priors_xyxy = xywha_to_xyxy(priors_xywha)
        gt_locations = list()
        gt_labels = list()
        for box, label in zip(targets['boxes'], targets['labels']):
            box, label = assign_priors(box, label, priors_xyxy, self.iou_threshold)
            locations = encode(box, priors_xywha, self.variances)
            gt_locations.append(locations)
            gt_labels.append(label)

        gt_locations = torch.stack(gt_locations, 0)
        gt_labels = torch.stack(gt_labels, 0)
        return gt_locations, gt_labels


class PriorBoxGenerator:
    """Generate priorbox coordinates in XYWHA_REL BoxMode for each source feature map
    """
    def __init__(self, cfg):
        cfg = config_parse(cfg)

        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def __call__(self):
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
        output = torch.tensor(priors)
        if self.clip:
            output.clamp_(max=1, min=0)

        return output
