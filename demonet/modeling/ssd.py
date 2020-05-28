import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_heads.multibox_head import MultiBoxHeads


model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(
        self, base, extras, head, **kwargs,
    ):
        super().__init__()

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.l2norm = nn.BatchNorm2d(512)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.multibox_heads = MultiBoxHeads(**kwargs)

    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        """Applies network layers and ops on input image(s).
        Args:
            images: input image or batch of images. Shape: [batch, 3, 300, 300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch, topk, 7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors, num_classes]
                    2: localization layers, Shape: [batch, num_priors * 4]
                    3: priorbox layers, Shape: [2, num_priors * 4]
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            images = self.vgg[k](images)

        s = self.l2norm(images)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            images = self.vgg[k](images)
        sources.append(images)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            images = F.relu(v(images), inplace=True)
            if k % 2 == 1:
                sources.append(images)

        # apply multibox head to source layers
        for s, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(s).permute(0, 2, 3, 1).contiguous())
            conf.append(c(s).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], dim=1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], dim=1)

        detections, detector_losses = self.multibox_heads(loc, conf, targets)

        return self.eager_outputs(detector_losses, detections)


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            if batch_norm:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            in_channels = v

    layers += [
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # pool5, retains size
        nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),  # conv6, atrous convolution
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, kernel_size=1),  # conv7
        nn.ReLU(inplace=True),
    ]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [
                    nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1),
                ]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]),
                ]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [
            nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1),
        ]
        conf_layers += [
            nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1),
        ]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [
            nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1),
        ]
        conf_layers += [
            nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1),
        ]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_model(size=300, num_classes=21, **kwargs):
    if size != 300:
        raise NotImplementedError(
            "You specified size [{}]. However, currently only "
            "SSD300 (size=300) is supported!".format(size),
        )

    base_layers, extras_layers, head_layers = multibox(
        vgg(base[str(size)], 3),
        add_extras(extras[str(size)], 1024),
        mbox[str(size)],
        num_classes,
    )

    model = SSD(
        base_layers,
        extras_layers,
        head_layers,
        image_size=size,
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        feature_maps=[38, 19, 10, 5, 3, 1],
        min_ratio=20,
        max_ratio=90,
        steps=[8, 16, 32, 64, 100, 300],
        clip=False,
        **kwargs,
    )

    return model
