# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.

from .backbone import build_backbone
from .prior_box import AnchorGenerator
from .box_head import MultiBoxLiteHead, PostProcess
from .generalized_ssd import GeneralizedSSD


class SSDLiteWithMobileNetV2(GeneralizedSSD):
    r"""MobileNet V2 SSD model class
    Args:
        backbone: MobileNet V2 backbone layers with extra layers
        head: "multibox head" consists of box_regression and class_logits conv layers
    """
    def __init__(
        self,
        backbone,
        # Anchor parameters
        image_size=320,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        min_sizes=[60, 105, 150, 195, 240, 285],
        max_sizes=[105, 150, 195, 240, 285, 330],
        clip=True,
        # Multi Box parameter
        hidden_dims=[96, 1280, 512, 256, 256, 64],
        num_anchors=[6, 6, 6, 6, 6, 6],  # number of boxes per feature map location
        num_classes=21,
        # Box post process
        score_thresh=0.5,
    ):
        prior_generator = AnchorGenerator(image_size, aspect_ratios, min_sizes, max_sizes, clip)
        multibox_head = MultiBoxLiteHead(hidden_dims, num_anchors, num_classes)
        post_process = PostProcess(score_thresh=score_thresh)

        super().__init__(backbone, prior_generator, multibox_head, post_process)


def build(args):
    backbone = build_backbone(args)

    model = SSDLiteWithMobileNetV2(
        backbone,
        image_size=args.image_size,
        num_classes=args.num_classes,
        score_thresh=args.score_thresh,
    )

    return model
