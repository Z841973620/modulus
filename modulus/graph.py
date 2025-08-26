""" Helper functions for unrolling computational graph
"""

from copy import copy
import torch
from typing import Dict, List


from .node import Node
from .key import Key
from .constants import diff_str
from .eq.derivatives import Derivative


class Graph(torch.nn.Module):
    """
    Torch Module that is constructed by unrolling a computational graph given
    desired inputs, outputs, and evaluatable nodes.

    Examples
    ========
    Here is a simple example of using `Graph` to unroll a two node graph.
    >>> import torch
    >>> from sympy import Symbol
    >>> from modulus.node import Node
    >>> from modulus.key import Key
    >>> from modulus.graph import Graph
    >>> node_1 = Node.from_sympy(Symbol('x') + Symbol('y'), 'u')
    >>> node_2 = Node.from_sympy(Symbol('u') + 1.0, 'v')
    >>> graph = Graph([node_1, node_2], [Key('x'), Key('y')], [Key('v')])
    >>> graph.forward({'x': torch.tensor([1.0]), 'y': torch.tensor([2.0])})
    {'v': tensor([4.])}

    Parameters
    ----------
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.
    invar : List[Key]
        List of inputs to graph.
    req_names : List[Key]
        List of required outputs of graph.
    diff_nodes : List[Node]
        List of specialty nodes to compute derivatives.
        By default this is not needed.
    """

    def __init__(
        self,
        nodes: List[Node],
        invar: List[Key],
        req_names: List[Key],
        diff_nodes: List[Node] = [],
        jit_autograd_nodes: bool = True,
    ):
        super().__init__()

        self.req_names = req_names
        self.computable_names = set(_computable_names(nodes, invar))

        # check if graph can be computed
        req_names_no_diff = [Key(x.name) for x in req_names]
        if not set(req_names_no_diff).issubset(self.computable_names):
            _print_graph_unroll_error(nodes, invar, req_names)
            raise RuntimeError("Failed Unrolling Graph")

        # compute only necessary nodes for req_names
        # Walk backwards from the output nodes in the graph and keep adding required inputs
        # until all inputs are available in invar
        nodes = copy(nodes)
        necessary_nodes = []
        needed_names = [Key(x.name, derivatives=x.derivatives) for x in req_names] + [
            Key(x.name) for x in req_names
        ]
        while True:
            finished = True
            for i, node in enumerate(nodes):
                if not set(node.outputs).isdisjoint(set(needed_names)):
                    # Make needed names include derivatives!
                    needed_names += (
                        node.inputs
                        + [
                            Key(x.name, derivatives=x.derivatives)
                            for x in node.derivatives
                        ]
                        + [Key(x.name) for x in node.derivatives]
                    )
                    # needed_names.update(node.inputs() + [Key(x.name) for x in node.derivatives()])
                    necessary_nodes.append(node)
                    nodes.pop(i)
                    finished = False
            if finished:
                break

        # unroll graph with only necessary nodes
        # Store node evaluation order to use at runtime
        self.node_evaluation_order = []
        outvar = copy(invar)
        while True:
            # compute all nodes that don't need derivative calls
            while True:
                finished = True
                for i, node in enumerate(necessary_nodes):
                    if set(node.inputs + node.derivatives).issubset(set(outvar)):
                        self.node_evaluation_order.append(node)
                        outvar += node.outputs
                        necessary_nodes.pop(i)
                        finished = False
                if finished:
                    break
            # compute derivative calls all at once
            needed_derivatives = []
            for node in necessary_nodes:
                needed_derivatives += node.derivatives
            needed_derivatives += [x for x in req_names if x.derivatives]
            needed_derivatives = [
                diff for diff in needed_derivatives if diff not in outvar
            ]  # remove already computed diffs
            if len(needed_derivatives) > 0:
                # check if solution in diff nodes
                try_auto_diff = True
                for dn in diff_nodes:
                    if (not set(dn.outputs).isdisjoint(set(needed_derivatives))) and (
                        set(dn.inputs).issubset(set(outvar))
                    ):
                        # input_variables = Variables.subset(outvar, dn.inputs())
                        # outvar.update(dn.evaluate(input_variables))
                        self.node_evaluation_order.append(dn)
                        outvar += dn.outputs
                        try_auto_diff = False

                # compute first derivatives only
                if try_auto_diff:
                    # Variables.differentiate(outvar, outvar, needed_derivatives)
                    dnode = Derivative.make_node(
                        outvar, needed_derivatives, jit=jit_autograd_nodes
                    )
                    self.node_evaluation_order.append(dnode)
                    outvar += dnode.outputs

            # check if finished
            if set(req_names).issubset(set(outvar)):
                # return Variables({key: value for key, value in outvar.items() if key in req_names})
                break

        self.evaluation_order = torch.nn.ModuleList(
            [n.evaluate for n in self.node_evaluation_order]
        )
        self.node_names: List[str] = [n.name for n in self.node_evaluation_order]
        self.optimizer_list = torch.nn.ModuleList(
            [n.evaluate for n in self.node_evaluation_order if n.optimize]
        )

    def forward(self, invar: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outvar = invar
        for i, e in enumerate(self.evaluation_order):
            torch.cuda.nvtx.range_push(self.node_names[i])
            outvar.update(e(outvar))
            torch.cuda.nvtx.range_pop()
        outvar = {
            key: value for key, value in outvar.items() if Key(key) in self.req_names
        }
        return outvar


def _print_graph_unroll_error(nodes, invar, req_names):
    print("####################################")
    print("could not unroll graph!")
    print(
        "This is probably because you are asking to compute a value that is not an output of any node"
    )
    print("####################################")
    print("invar: " + str(list(invar)))
    print("requested var: " + str(req_names))
    print("computable var: " + str(_computable_names(nodes, invar)))
    print("####################################")
    print("Nodes in graph: ")
    for node in nodes:
        print(node)
    print("####################################")


def _computable_names(nodes, invar):
    nodes = copy(nodes)
    computable_names = copy(invar)
    while True:
        finished = True
        for i, node in enumerate(nodes):
            if set(node.inputs).issubset(set(computable_names)):
                computable_names += node.outputs
                nodes.pop(i)
                finished = False
        if finished:
            return computable_names
