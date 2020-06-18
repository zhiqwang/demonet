import torch

from models.ssd_mobilenet import SSDLiteWithMobileNetV2, build_backbone, build_extras, build_multibox

dependencies = ["torch", "torchvision"]


def _make_mobilenet_v2(num_classes, image_size, score_thresh=0.5):
    backbone = build_backbone(train_backbone=False)
    extras_layers = build_extras(backbone.num_channels)
    anchor_nms_cfg = [6, 6, 6, 6, 6, 6]  # number of boxes per feature map location
    head_layers = build_multibox(anchor_nms_cfg, num_classes)

    model = SSDLiteWithMobileNetV2(
        backbone,
        extras_layers,
        head_layers,
        score_thresh=score_thresh,
        image_size=image_size,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
        feature_maps=[19, 10, 5, 3, 2, 1],
        min_ratio=20,
        max_ratio=80,
        steps=[16, 32, 64, 100, 150, 300],
        clip=True,
        min_sizes=[60, 105, 150, 195, 240, 285],
        max_sizes=[105, 150, 195, 240, 285, 330],
    )

    return model


model_urls = {'ssd_lite_mobilenet_v2': './checkpoints/mobilenet_v2/ssd_lite_mobilenet_v2_199.pth'}


def ssd_lite_mobilenet_v2(
    pretrained=False,
    num_classes=21,
    image_size=300,
    score_thresh=0.5,
    return_postprocessor=False,
):
    """
    ssd lite with mobilenet v2 backbone.
    Achieves 68.39 AP50 on PASCAL VOC.
    """
    model = _make_mobilenet_v2(
        num_classes=num_classes,
        image_size=image_size,
        score_thresh=score_thresh,
    )
    if pretrained:
        checkpoint = torch.load(model_urls['ssd_lite_mobilenet_v2'], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
