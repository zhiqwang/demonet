# Copyright (c) 2020, Zhiqiang Wang. All Rights Reserved.

from .generalized_ssd import GeneralizedSSD
from .backbone import build_backbone
from .prior_box import AnchorGenerator
from .box_head import MultiBoxLiteHead, SSDBoxHeads


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
        hidden_dims=[576, 1280, 512, 256, 256, 64],
        num_anchors=[6, 6, 6, 6, 6, 6],  # number of boxes per feature map location
        num_classes=21,
        # SSD Box parameter
        variances=[0.1, 0.2],
        iou_thresh=0.5,
        negative_positive_ratio=3,
        score_thresh=0.5,
        nms_thresh=0.45,
        post_nms_top_n=100,
    ):

        prior_generator = AnchorGenerator(image_size, aspect_ratios, min_sizes, max_sizes, clip)

        multibox_head = MultiBoxLiteHead(hidden_dims, num_anchors, num_classes)

        ssd_box_heads = SSDBoxHeads(
            prior_generator, multibox_head,
            variances, iou_thresh, negative_positive_ratio,
            score_thresh, nms_thresh, post_nms_top_n,
        )

        super().__init__(backbone, ssd_box_heads)


model_urls = {'ssd_lite_mobilenet_v2': ''}


def build(args):
    if args.image_size != 300:
        raise NotImplementedError(
            "You specified image_size [{}]. However, currently only "
            "MobileNetV2SSD300 (image_size=300) is supported!".format(args.image_size),
        )

    backbone = build_backbone(args)

    model = SSDLiteWithMobileNetV2(
        backbone,
        image_size=args.image_size,
        score_thresh=args.score_thresh,
        num_classes=args.num_classes,
    )

    return model
