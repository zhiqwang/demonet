"""
Plotting a graph with bad gradient nodes marked in red.
Mostly copy-paste from https://github.com/t-vi/pytorch-tvmisc/tree/master/visualize
"""
from graphviz import Digraph
import numpy as np
import tvm


def collect_ops(node):
    ops = set()

    def visitor(e):
        if isinstance(e, tvm.ir.Op):
            ops.add(e.name)

    tvm.relay.analysis.post_order_visit(node, visitor)
    return ops


def to_str(node):
    if isinstance(node, tvm.relay.Constant):
        return repr(node).lstrip('Constant(')[:-1]
    else:
        raise NotImplementedError("to_str:" + repr(node))


def is_small_const(const, collapse_small=True):
    if not (collapse_small and isinstance(const, tvm.relay.Constant)):
        return False
    if isinstance(const.data, tvm.runtime.ndarray.NDArray):
        return np.prod(const.data.shape) < 10
    return True


def visualize(expr, collapse_small=True, node_attr_dict={}):

    # node_dict maps a Relay node to an index (node ID)
    def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        node_dict[node] = len(node_dict)

    node_dict = {}
    tvm.relay.analysis.post_order_visit(expr, lambda x: _traverse_expr(x, node_dict))

    dot = Digraph(format='svg', graph_attr=dict(size="8"))
    dot.attr('node', shape='box')

    # Sort by node ID
    for node, node_id in sorted(node_dict.items(), key=lambda x: x[1]):
        if isinstance(node, tvm.relay.Function):
            dot.node(str(node_id), 'Function', **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.body]), str(node_id))
        elif isinstance(node, tvm.relay.Var):
            if node.type_annotation is not None:
                if hasattr(node.type_annotation, 'shape'):
                    shape = tuple([int(x) for x in node.type_annotation.shape])
                    dtype = node.type_annotation.dtype
                    typstr = f'Tensor[{shape}, {dtype}]'
                else:
                    typstr = str(node.type_annotation)
            else:
                typstr = '?'
            d = dict(shape='ellipse')
            d.update(node_attr_dict.get(node, {}))
            dot.node(str(node_id), f'{node.name_hint}: {typstr}', **d)
        elif isinstance(node, tvm.relay.Tuple):
            dot.node(str(node_id), 'Tuple[...])', **node_attr_dict.get(node, {}))
            for field in node.fields:
                dot.edge(str(node_dict[field]), str(node_id))
        elif isinstance(node, tvm.relay.Constant):
            if not is_small_const(node, collapse_small=collapse_small):  # small consts are shown in ops
                dot.node(
                    str(node_id),
                    f'Constant({node.data.shape}, {node.data.dtype})',
                    **node_attr_dict.get(node, {}),
                )
        elif isinstance(node, tvm.relay.Call):
            node_name, args_with_edge = visualize_call(node, node_id, collapse_small, node_attr_dict)

            dot.node(str(node_id), node_name, **node_attr_dict.get(node, {}))
            for arg in args_with_edge:
                dot.edge(str(node_dict[arg]), str(node_id))
        elif isinstance(node, tvm.ir.Op):
            # dot.node(str(node_id), f'Op {node.name}')
            pass  # covered in call
        elif isinstance(node, tvm.relay.TupleGetItem):
            dot.node(str(node_id), f'TupleGetItem(idx={node.index})', **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.tuple_value]), str(node_id))
        elif isinstance(node, tvm.relay.Let):
            dot.node(str(node_id), 'Let(XX)', **node_attr_dict.get(node, {}))
            dot.edge(str(node_dict[node.value]), str(node_id))
            dot.edge(str(node_id), str(node_dict[node.var]))
        else:
            raise RuntimeError(f'Unknown node type. node_id: {node_id}, node: {type(node)}')

    return dot


def visualize_call(node, node_id, collapse_small, node_attr_dict):
    args_with_edge = []
    arg_str_list = []
    for arg in node.args:
        if is_small_const(arg, collapse_small=collapse_small):
            arg_str_list.append(to_str(arg))
        else:
            arg_str_list.append('·')
            args_with_edge.append(arg)
    arg_str = ', '.join(arg_str_list)
    if isinstance(node.op, tvm.ir.Op):
        name = node.op.name
        attrs = {k: getattr(node.attrs, k) for k in node.attrs.keys()} if hasattr(node.attrs, 'keys') else {}
        # attrs = inspect.getmembers(node.attrs)
        attr_str_list = [k + '=' + (str(v) if len(str(v)) < 20 else "...") for k, v in attrs.items()]
        if attr_str_list:
            attr_str = '| ' + ', '.join(attr_str_list)
        else:
            attr_str = ''
    else:
        ops = collect_ops(node)
        if ops:
            name = '_'.join(ops)
        else:
            name = '...'
        attr_str = ''
    node_name = f'{name}({arg_str}{attr_str})'

    return node_name, args_with_edge


def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)


def register_hooks(var):
    fn_dict = {}

    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return grad_output.isnan().any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(
            style='filled',
            shape='box',
            align='left',
            fontsize='12',
            ranksep='0.1',
            height='0.2',
        )
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '(' + (', ').join(map(str, size)) + ')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot
