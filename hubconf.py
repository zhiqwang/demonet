import torch

from models.backbone import MobileNetWithExtraBlocks
from models.ssd_mobilenet import SSDLiteWithMobileNetV2
from models.generalized_ssd import PostProcess

dependencies = ["torch", "torchvision"]


def _make_mobilenet_v2(image_size=320, score_thresh=0.5, num_classes=21):
    backbone_with_extra_blocks = MobileNetWithExtraBlocks(train_backbone=True)

    model = SSDLiteWithMobileNetV2(
        backbone_with_extra_blocks,
        image_size=image_size,
    )

    return model


model_urls = {'ssd_lite_mobilenet_v2': './checkpoints/mobilenet_v2/ssd_lite_mobilenet_v2_199.pth'}


def ssd_lite_mobilenet_v2(
    pretrained=False,
    image_size=320,
    score_thresh=0.5,
    num_classes=21,
    return_postprocessor=False,
):
    """
    ssd lite with mobilenet v2 backbone.
    Achieves 68.39 AP50 on PASCAL VOC.
    """
    model = _make_mobilenet_v2(image_size=image_size)
    if pretrained:
        checkpoint = torch.load(model_urls['ssd_lite_mobilenet_v2'], map_location="cpu")
        model.load_state_dict(checkpoint)
    if return_postprocessor:
        postprocessor = PostProcess(score_thresh=score_thresh)
        return model, postprocessor
    return model
