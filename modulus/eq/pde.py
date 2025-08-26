""" base class for PDEs
"""

from sympy import (
    Symbol,
    Function,
    init_printing,
    pprint,
    latex,
    preview,
    Matrix,
    Eq,
    Basic,
)

from modulus.node import Node
from modulus.constants import diff_str
from modulus.key import Key


class PDE(object):
    """base class for all partial differential equations"""

    name = "PDE"

    def __init__(self):
        super().__init__()
        self.equations = Variables()

    def pprint(self, print_latex=False):
        """
        Print differential equation.

        Parameters
        ----------
        print_latex : bool
            If True print the equations in Latex. Else, just
            print as text.
        """
        init_printing(use_latex=True)
        for key, value in self.equations.items():
            print(str(key) + ": " + str(value))
        if print_latex:
            preview(
                Matrix(
                    [
                        Eq(Function(name, real=True), eq)
                        for name, eq in self.equations.items()
                    ]
                ),
                mat_str="cases",
                mat_delim="",
            )

    def subs(self, x, y):
        for name, eq in self.equations.items():
            self.equations[name] = eq.subs(x, y).doit()

    def make_nodes(self, detach_names=[]):
        """
        Make a list of nodes from PDE.

        Parameters
        ----------
        detach_names : List[str]
            This will detach the inputs of the resulting node.

        Returns
        -------
        nodes : List[Node]
            Makes a separate node for every equation.
        """
        nodes = []
        for name, eq in self.equations.items():
            nodes.append(Node.from_sympy(eq, str(name), detach_names))
        return nodes
