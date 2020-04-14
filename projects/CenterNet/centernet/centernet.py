import torch
import torch.nn as nn

from .backbone_utils import Backbone
from .deconv import DeconvLayers
from .heads import CtdetHeads
from .transform import CtdetTransform


__all__ = [
    'centernet_resnet18',
    'centernet_mobilenetv2',
    'centernet_shufflenetv2',
]


class CenterNet(nn.Module):
    def __init__(
        self,
        backbone_name,
        heads,
        head_conv,
        feature_name='layer4',
        use_deconv=True,
        pretrained=True,
        decode_mode='1d',
    ):
        super().__init__()
        # image encode and decode
        self.transform = CtdetTransform(decode_mode=decode_mode)

        self.backbone = Backbone(
            backbone_name,
            feature_name=feature_name,
            pretrained=pretrained,
        )

        # used for deconv layers
        if feature_name == 'layer4':
            deconv_inplanes = 512
        elif feature_name == 'features':
            deconv_inplanes = 1280
        elif feature_name == 'conv5':
            deconv_inplanes = 1024
        else:
            deconv_inplanes = 0

        deconv_with_bias = False
        self.deconv_layers = DeconvLayers(
            3,
            [256, 256, 256],
            [4, 4, 4],
            inplanes=deconv_inplanes,
            deconv_with_bias=deconv_with_bias,
        ) if use_deconv else None

        inplanes = 256 if use_deconv else deconv_inplanes
        self.heads = CtdetHeads(
            heads,
            head_conv,
            inplanes=inplanes,
        )

    @torch.jit.unused
    def eager_outputs(self, detections, losses):
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        features = self.backbone(images)
        if self.deconv_layers is not None:
            features = self.deconv_layers(features)

        detections, losses = self.heads(features, targets=targets)

        detections = self.transform.postprocess(detections)

        return self.eager_outputs(detections, losses)


def centernet_resnet18(
    heads,
    head_conv=64,
    feature_name='layer4',
    use_deconv=False,
    pretrained=True,
    decode_mode='1d',
):
    model = CenterNet(
        backbone_name='resnet18',
        heads=heads,
        head_conv=head_conv,
        feature_name=feature_name,
        use_deconv=use_deconv,
        pretrained=pretrained,
        decode_mode=decode_mode,
    )

    return model


def centernet_mobilenetv2(
    heads,
    head_conv=64,
    feature_name='features',
    use_deconv=False,
    pretrained=True,
    decode_mode='1d',
):
    model = CenterNet(
        backbone_name='mobilenet_v2',
        heads=heads,
        head_conv=head_conv,
        feature_name=feature_name,
        use_deconv=use_deconv,
        pretrained=pretrained,
        decode_mode=decode_mode,
    )

    return model


def centernet_shufflenetv2(
    heads,
    head_conv=64,
    feature_name='conv5',
    use_deconv=False,
    pretrained=True,
    decode_mode='1d',
):
    model = CenterNet(
        backbone_name='shufflenet_v2_x0_5',
        heads=heads,
        head_conv=head_conv,
        feature_name=feature_name,
        use_deconv=use_deconv,
        pretrained=pretrained,
        decode_mode=decode_mode,
    )

    return model
