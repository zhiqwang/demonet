from .engine.hooks import Hook


def parse(model, input, model_name='TransferedPytorchModel'):
    print('>>> Starting transform, this will take a while...')
    hooks = {}
    for name, module in model.named_modules():
        hooks[name] = Hook(module)

    _ = model(input)

    for key, value in hooks.items():
        print('Key: \'{}\'\nInput, length: {}, the first shape: {}\nOutput: {}'.format(
            key, len(hooks[key].input), hooks[key].input[0].shape,
            hooks[key].output.shape,
        ))
