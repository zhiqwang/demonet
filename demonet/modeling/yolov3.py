import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import parse_model_cfg, save_weights, load_darknet_weights
from demonet.utils import torch_utils

ONNX_EXPORT = False


def create_modules(module_defs, img_size, arch):
    # Constructs module list of layer blocks from module configuration in module_defs

    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layes
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (
                int(mdef['stride_y']), int(mdef['stride_x']),
            )
            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module(
                'Conv2d',
                nn.Conv2d(in_channels=output_filters[-1],
                          out_channels=filters,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=pad,
                          bias=not bn),
            )
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            # TODO: activation study https://github.com/ultralytics/yolov3/issues/441
            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                # modules.add_module('activation', Swish())

        elif mdef['type'] == 'maxpool':
            kernel_size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) // 2),
            )
            if kernel_size == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + i for l in layers])
            # if mdef[i+1]['type'] == 'reorg3d':
            #     modules = nn.Upsample(
            #         scale_factor=1/float(mdef[i+1]['stride']),
            #         mode='nearest',  # reorg3d
            #     )

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            # torch.Size([16, 128, 104, 104])
            # torch.Size([16, 64, 208, 208]) <-- # stride 2 interpolate
            # dimensions 2 and 3 to cat with prior layer
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')]  # anchor mask
            modules = YOLOLayer(
                anchors=mdef['anchors'][mask],  # anchor list
                nc=int(mdef['classes']),  # number of classes
                img_size=img_size,  # (416, 416)
                yolo_index=yolo_index,  # 0, 1 or 2
                arch=arch,  # yolo architecture
            )

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            if arch == 'defaultpw' or arch == 'Fdefaultpw':  # default with positive weights
                b = [-4, -3.6]  # obj, cls
            elif arch == 'default':  # default no pw (40 cls, 80 obj)
                b = [-5.5, -4.0]
            elif arch == 'uBCE':  # unified BCE (80 classes)
                b = [0, -8.5]
            elif arch == 'uCE':  # unified CE (1 background + 80 classes)
                b = [10, -0.1]
            elif arch == 'Fdefault':  # Focal default no pw (28 cls, 21 obj, no pw)
                b = [-2.1, -1.8]
            elif arch == 'uFBCE' or arch == 'uFBCEpw':  # unified FocalBCE (5120 obj, 80 classes)
                b = [0, -6.5]
            elif arch == 'uFCE':  # unified FocalCE (64 cls, 1 background + 80 classes)
                b = [7.7, -1.1]

            bias = module_list[-1][0].bias.view(len(mask), -1)  # 255 to 3x85
            bias[:, 4] += b[0] - bias[:, 4].mean()  # obj
            bias[:, 5:] += b[1] - bias[:, 5:].mean()  # cls
            # bias = torch.load('weights/yolov3-spp.bias.pt')[yolo_index]  # list of tensors [3x85, 3x85, 3x85]
            module_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))
            # utils.print_model_biases(model)

        else:
            print('Warning: Unrecognized Layer Type: {}'.format(mdef['type']))

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, arch):
        super().__init__()

        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.nx = 0  # initialize number of x gridpoints
        self.ny = 0  # initialize number of y gridpoints
        self.arch = arch

        if ONNX_EXPORT:  # grids must be computed in __init__
            stride = [32, 16, 8][yolo_index]  # stride of this layer
            nx = int(img_size[1] / stride)  # number x grid points
            ny = int(img_size[0] / stride)  # number y grid points
            create_grids(self, img_size, (nx, ny))

    def forward(self, preds, img_size, var=None):
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            bs, ny, nx = preds.shape[0], preds.shape[-2], preds.shape[-1]
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), preds.device, preds.dtype)

        # preds.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 85)
        # (bs, anchors, grid, grid, classes + xywh)
        preds = preds.view(
            bs, self.na, self.nc + 5, self.ny, self.nx,
        ).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return preds

        elif ONNX_EXPORT:
            # Constants CAN NOT BE BROADCAST, ensure correct shape!
            ngu = self.ng.repeat((1, self.na * self.nx * self.ny, 1))
            grid_xy = self.grid_xy.repeat((1, self.na, 1, 1, 1)).view((1, -1, 2))
            anchor_wh = self.anchor_wh.repeat((
                1, 1, self.nx, self.ny, 1,
            )).view((1, -1, 2)) / ngu

            preds = preds.view(-1, 5 + self.nc)
            xy = torch.sigmoid(preds[..., 0:2]) + grid_xy[0]  # x, y
            wh = torch.exp(preds[..., 2:4]) * anchor_wh[0]  # width, height
            p_conf = torch.sigmoid(preds[:, 4:5])  # Conf
            p_cls = F.softmax(preds[:, 5:85], 1) * p_conf  # SSD-like conf
            return torch.cat((xy / ngu[0], wh, p_conf, p_cls), 1).t()

            # preds = preds.view(1, -1, 5 + self.nc)
            # xy = torch.sigmoid(preds[..., 0:2]) + grid_xy  # x, y
            # wh = torch.exp(preds[..., 2:4]) * anchor_wh  # width, height
            # p_conf = torch.sigmoid(preds[..., 4:5])  # Conf
            # p_cls = preds[..., 5:5 + self.nc]
            # # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
            # # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
            # p_cls = torch.exp(p_cls).permute((2, 1, 0))
            # p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
            # p_cls = p_cls.permute(2, 1, 0)
            # return torch.cat((xy / ngu, wh, p_conf, p_cls), 2).squeeze().t()

        else:  # inference
            # s = 1.5  # scale_xy  (pxy = pxy * s - (s - 1) / 2)
            io = preds.clone()  # inference output
            io[..., 0:2] = torch.sigmoid(io[..., 0:2]) + self.grid_xy  # xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
            # io[..., 2:4] = ((torch.sigmoid(io[..., 2:4]) * 2) ** 3) * self.anchor_wh  # wh power method
            io[..., :4] *= self.stride

            if 'default' in self.arch:  # seperate obj and cls
                torch.sigmoid_(io[..., 4:])
            elif 'BCE' in self.arch:  # unified BCE (80 classes)
                torch.sigmoid_(io[..., 5:])
                io[..., 4] = 1
            elif 'CE' in self.arch:  # unified CE (1 background + 80 classes)
                io[..., 4:] = F.softmax(io[..., 4:], dim=4)
                io[..., 4] = 1

            if self.nc == 1:
                io[..., 5] = 1  # single-class model https://github.com/ultralytics/yolov3/issues/235

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return io.view(bs, -1, 5 + self.nc), preds


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), arch='default'):
        super().__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, arch)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training

    def forward(self, x, var=None):
        img_size = x.shape[-2:]
        layer_outputs = []
        output = []

        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    if layer_outputs[layers[0]].shape[-2:] == layer_outputs[layers[1]].shape[-2:]:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    else:  # apply stride 2 for darknet reorg layer
                        # print(('{} ' * len(layers)).format(*[layer_outputs[i].shape for i in layers]))
                        layer_outputs[layers[1]] = F.interpolate(
                            layer_outputs[layers[1]],
                            scale_factor=[0.5, 0.5],
                        )
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]
            elif mtype == 'yolo':
                x = module(x, img_size)
                output.append(x)
            layer_outputs.append(x if i in self.routs else [])

        if self.training:
            return output
        elif ONNX_EXPORT:
            output = torch.cat(output, 1)  # cat 3 layers 85 x (507, 2028, 8112) to 85 x 10647
            nc = self.module_list[self.yolo_layers[0]].nc  # number of classes
            return output[5:5 + nc].t(), output[:4].t()  # ONNX scores, boxes
        else:
            io, preds = list(zip(*output))  # inference output, training output
            return torch.cat(io, 1), preds

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        # model_info(self)  # yolov3-spp reduced from 225 to 152 layers


def get_yolo_layers(model):
    return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3


def create_grids(
    module, img_size=(416, 416), ng=(13, 13),
    device='cpu', type=torch.float32,
):
    nx, ny = ng  # x and y grid size
    module.img_size = max(img_size)
    module.stride = module.img_size / max(ng)

    # build xy offsets
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    module.grid_xy = torch.stack((xv, yv), 2).to(device).type(type).view((1, 1, ny, nx, 2))

    # build wh gains
    module.anchor_vec = module.anchors.to(device) / module.stride
    module.anchor_wh = module.anchor_vec.view(1, module.na, 1, 1, 2).to(device).type(type)
    module.ng = torch.Tensor(ng).to(device)
    module.nx = nx
    module.ny = ny


def convert(cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights'):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # from models import *; convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)

    # Load weights and save
    if weights.endswith('.pt'):  # if PyTorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        save_weights(model, path='converted.weights', cutoff=-1)
        print("Success: converted '%s' to 'converted.weights'" % weights)

    elif weights.endswith('.weights'):  # darknet format
        _ = load_darknet_weights(model, weights)

        chkpt = {'epoch': -1,
                 'best_fitness': None,
                 'training_results': None,
                 'model': model.state_dict(),
                 'optimizer': None}

        torch.save(chkpt, 'converted.pt')
        print("Success: converted '%s' to 'converted.pt'" % weights)

    else:
        print('Error: extension not supported.')
