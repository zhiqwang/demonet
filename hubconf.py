import torch

from models.backbone import BackboneWithMobileNet, ExtraLayers, Joiner
from models.ssd_mobilenet import SSDLiteWithMobileNetV2

dependencies = ["torch", "torchvision"]


def _make_mobilenet_v2(image_size=300, score_thresh=0.5, num_classes=21):
    backbone = BackboneWithMobileNet(train_backbone=True)
    extra_layers = ExtraLayers(backbone.num_channels)
    backbone_with_extra_layers = Joiner(backbone, extra_layers)

    model = SSDLiteWithMobileNetV2(
        backbone_with_extra_layers,
        image_size=image_size,
        score_thresh=score_thresh,
        num_classes=num_classes,
    )

    return model


model_urls = {'ssd_lite_mobilenet_v2': './checkpoints/mobilenet_v2/ssd_lite_mobilenet_v2_199.pth'}


def ssd_lite_mobilenet_v2(
    pretrained=False,
    image_size=300,
    score_thresh=0.5,
    num_classes=21,
    return_postprocessor=False,
):
    """
    ssd lite with mobilenet v2 backbone.
    Achieves 68.39 AP50 on PASCAL VOC.
    """
    model = _make_mobilenet_v2(
        image_size=image_size,
        score_thresh=score_thresh,
        num_classes=num_classes,
    )
    if pretrained:
        checkpoint = torch.load(model_urls['ssd_lite_mobilenet_v2'], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model
