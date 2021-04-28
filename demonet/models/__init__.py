from .ssd_mobilenet import build as build_mobilenet_v2_ssd_lite


def build_model(args):
    if args.arch == 'ssd_lite_mobilenet_v2':
        return build_mobilenet_v2_ssd_lite(args)

    if args.arch == 'pelee':
        from .pelee import build as build_pelee
        return build_pelee(args)

    raise ValueError(f'model {args.arch} not supported')
