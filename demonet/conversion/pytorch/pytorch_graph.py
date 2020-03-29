import re

import torch
import torch.jit
from torch.jit import _unique_state_dict
import torch.onnx.utils
from torch.onnx import OperatorExportTypes

from demonet.conversion.common.graph import GraphNode, Graph


class PytorchGraphNode(GraphNode):

    def __init__(self, layer):
        self._name = layer.scopeName()
        self._kind = layer.kind()
        node_id = re.search(r"[\d]+", layer.__str__())
        self.id = node_id.group(0)

        super().__init__(layer)
        if "L2Norm" not in self.name:
            self.attrs = {k : layer[k] for k in layer.attributeNames()}

        self.weights_name = '.'.join(
            re.findall(r'\[([\w\d.]+)\]', self._name)
        )

    @property
    def name(self):
        name = self._name + self.id
        # Scopes created in a nested scope may have initial characters
        # that are illegal as the initial character of an op name
        # (viz. '-', '\', '/', and '_').
        name = name.replace('-', 'n').replace('\\', 'n').replace('/', 'n')
        name = name.replace('_', 'n').replace('[', 'n').replace(']', 'n')
        return name

    @property
    def type(self):
        return self._kind

    @property
    def pytorch_layer(self):
        return self.layer


class PytorchGraph(Graph):

    def __init__(self, model):
        # sanity check.
        super().__init__(model)
        self.model = model
        self.state_dict = _unique_state_dict(self.model)
        self.shape_dict = dict()

    @staticmethod
    def get_node_id(node):
        node_id = re.search(r"[\d]+", node.__str__())
        return node_id.group(0)

    def build(self, shape):
        """
        build graph for pytorch
        """
        # construct graph
        dummy_input = torch.randn(shape, requires_grad=False)

        graph = torch.onnx.utils._trace(self.model, dummy_input, OperatorExportTypes.RAW)
        torch._C._jit_pass_lint(graph)
        # nodes
        nodes = list(graph.nodes())

        # input layer
        # TODO

        # build each layer
        flag_l2norm = False

        for node in nodes:

            if "L2Norm" in node.__str__():
                if 'SSDnL2NormnL2Normn96' in self.shape_dict.keys():  # exist
                    continue

                for k in self.shape_dict.keys():
                    node_id = self.get_node_id(node)
                    if str(int(node_id) - 1) in k:
                        output_shape = self.shape_dict[k]
                        node_input_name = k

                node_scope = node.scopeName()
                node_name = node_scope + node_id
                node_name = node_name.replace('-', 'n').replace('\\', 'n').replace('/', 'n')
                node_name = node_name.replace('_', 'n').replace('[', 'n').replace(']', 'n')

                self.shape_dict[node_name] = output_shape
                self.layer_map[node_name] = PytorchGraphNode(node)
                self.layer_name_map[node_name] = node_name

                self._make_connection(node_input_name, node_name)
                flag_l2norm = True
            else:
                node_id = self.get_node_id(node)
                node_scope = node.scopeName()
                node_name = node_scope + node_id
                node_name = node_name.replace('-', 'n').replace('\\', 'n').replace('/', 'n')
                node_name = node_name.replace('_', 'n').replace('[', 'n').replace(']', 'n')
                if 'Dynamic' in node.__str__():
                    output_shape = [0, 0, 0, 0]
                else:
                    output_shape_str = re.findall(r'[^()!]+', node.__str__())[1]
                    print('output shape: [{}]'.format(output_shape_str))
                    output_shape = [int(x.replace('!', '')) for x in output_shape_str.split(',')]

                self.shape_dict[node_name] = output_shape
                self.layer_map[node_name] = PytorchGraphNode(node)
                self.layer_name_map[node_name] = node_name

                # input
                if flag_l2norm:
                    self._make_connection('SSDnSequentialnvggnfrontnnReLUn22n95', node_name)
                    flag_l2norm = False
                    continue

                for node_input in list(node.inputs()):
                    if self.get_node_id(node_input.node()) == '107':
                        self._make_connection('SSDnL2NormnL2Normn96', node_name)
                        continue

                for node_input in list(node.inputs()):

                    if self.get_node_id(node_input.node()) and node_input.node().scopeName():
                        node_input_name = node_input.node().scopeName() + self.get_node_id(node_input.node())
                        node_input_name = node_input_name.replace('-', 'n').replace('\\', 'n').replace('/', 'n')
                        node_input_name = node_input_name.replace('_', 'n').replace('[', 'n').replace(']', 'n')
                        self._make_connection(node_input_name, node_name)

        super().build()
