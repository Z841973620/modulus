"""Advection diffusion equation
Reference:
https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation
"""
from sympy import Symbol, Function, Number

from modulus.eq.pde import PDE
from modulus.node import Node


class AdvectionDiffusion(PDE):
    """
    Advection diffusion equation

    Parameters
    ==========
    T : str
        The dependent variable.
    D : float, Sympy Symbol/Expr, str
        Diffusivity. If `D` is a str then it is
        converted to Sympy Function of form 'D(x,y,z,t)'.
        If 'D' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    Q : float, Sympy Symbol/Expr, str
        The source term. If `Q` is a str then it is
        converted to Sympy Function of form 'Q(x,y,z,t)'.
        If 'Q' is a Sympy Symbol or Expression then this
        is substituted into the equation. Default is 0.
    rho : float, Sympy Symbol/Expr, str
        The density. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes.
    dim : int
        Dimension of the diffusion equation (1, 2, or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is False.

    Examples
    ========
    >>> ad = AdvectionDiffusion(D=0.1, rho=1.)
    >>> ad.pprint()
      advection_diffusion: u*T__x + v*T__y + w*T__z - 0.1*T__x__x - 0.1*T__y__y - 0.1*T__z__z
    >>> ad = AdvectionDiffusion(D='D', rho=1, dim=2, time=True)
    >>> ad.pprint()
      advection_diffusion: -D*T__x__x - D*T__y__y + u*T__x + v*T__y - D__x*T__x - D__y*T__y + T__t
    """

    name = "AdvectionDiffusion"

    def __init__(self, T="T", D="D", Q=0, rho="rho", dim=3, time=False):
        # set params
        self.T = T
        self.dim = dim
        self.time = time

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)

        # Temperature
        assert type(T) == str, "T needs to be string"
        T = Function(T)(*input_variables)

        # Diffusivity
        if type(D) is str:
            D = Function(D)(*input_variables)
        elif type(D) in [float, int]:
            D = Number(D)

        # Source
        if type(Q) is str:
            Q = Function(Q)(*input_variables)
        elif type(Q) in [float, int]:
            Q = Number(Q)

        # Density
        if type(rho) is str:
            rho = Function(rho)(*input_variables)
        elif type(rho) in [float, int]:
            rho = Number(rho)

        # curl
        curl = Number(0) if rho.diff() == 0 else u.diff(x) + v.diff(y) + w.diff(z)

        # set equations
        self.equations = {}
        advection = u * (T.diff(x)) + v * (T.diff(y)) + w * (T.diff(z))
        diffusion = (
            (D * T.diff(x)).diff(x) + (D * T.diff(y)).diff(y) + (D * T.diff(z)).diff(z)
        )
        self.equations["advection_diffusion"] = (
            T.diff(t) + advection + T * curl - diffusion
        )
