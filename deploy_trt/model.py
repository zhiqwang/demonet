# This class contains converted (ONNX) model metadata
class ModelData(object):
    # Name of input node
    INPUT_NAME = 'Input'
    # CHW format of model input
    INPUT_SHAPE = (3, 256, 256)
    # Name of output node
    OUTPUT_NAME = ['Hm', 'Shape', 'Offset']

    @staticmethod
    def get_input_channels():
        return ModelData.INPUT_SHAPE[0]

    @staticmethod
    def get_input_height():
        return ModelData.INPUT_SHAPE[1]

    @staticmethod
    def get_input_width():
        return ModelData.INPUT_SHAPE[2]
