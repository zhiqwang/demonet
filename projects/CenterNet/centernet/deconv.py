import torch.nn as nn


class DeconvLayers(nn.Module):
    def __init__(
        self, num_layers, num_filters, num_kernels,
        inplanes=512, deconv_with_bias=False,
    ):
        super().__init__()
        assert num_layers == len(num_filters), (
            'ERROR: num_deconv_layers is different len(num_deconv_filters)',
        )
        assert num_layers == len(num_kernels), (
            'ERROR: num_deconv_layers is different len(num_deconv_filters)',
        )

        self.inplanes = inplanes
        self.deconv_with_bias = deconv_with_bias

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        self.layers = nn.Sequential(*layers)

        for _, m in self.layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
