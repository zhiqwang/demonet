from .pelee import build as build_pelee


def build_model(args):
    if args.arch == 'pelee':
        return build_pelee(args)

    if args.arch == 'ssd_lite_mobilenet_v2':
        from .ssd_mobilenet import build as build_mobilenet_v2_ssd_lite
        return build_mobilenet_v2_ssd_lite(args)

    raise ValueError(f'model {args.arch} not supported')
