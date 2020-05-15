# Utility functions for building/saving/loading TensorRT Engine

import tensorrt as trt


def build_engine(
    model_path,
    trt_logger,
    max_workspace=2 << 30,
    fp16=False,
):
    with trt.Builder(trt_logger) as builder:
        builder.fp16_mode = True

        config = builder.create_builder_config()
        config.max_workspace_size = max_workspace
        if fp16:
            config.flags |= 1 << int(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        config.add_optimization_profile(profile)

        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        network = builder.create_network(explicit_batch)
        with open(model_path, 'rb') as model_fh:
            model = model_fh.read()

        with trt.OnnxParser(network, trt_logger) as parser:
            parser.parse(model)
            # last_layer = network.get_layer(network.num_layers - 1)
            # print('last layer: {}'.format(last_layer))
            # network.mark_output(last_layer.get_output(0))
            return builder.build_engine(network, config=config)


def save_engine(engine, engine_path):
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())


def load_engine(engine_path, trt_logger):
    with open(engine_path, 'rb') as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine
