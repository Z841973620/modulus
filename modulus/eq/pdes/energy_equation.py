"""Energy equation
Reference:
https://www.comsol.com/multiphysics/heat-transfer-conservation-of-energy
http://dl.icdst.org/pdfs/files1/2fe68e957cdf09a4862088ed279f00b0.pdf
http://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node14.html#e4.67
"""

from sympy import Symbol, Function
from sympy import *
from modulus.eq.pde import PDE
from ..constants import diff


class EnergyFluid(PDE):  # TODO clean function simlar to others
    """
    Energy equation
    Supports compressible flow.
    For Ideal gases only (uses the assumption that beta*T = 1).
    No heat/energy source added.

    Parameters
    ==========
    cp : str
        The specific heat.
    kappa : str
        The conductivity.
    rho : Sympy Symbol/Expr, str
        The density. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    nu : Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form 'nu(x,y,z,t)'.
        If 'nu' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    visc_heating : bool
        If viscous heating is applied or not. Default is False.
    dim : int
        Dimension of the energy equation (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is False.

    Examples
    ========
    >>> ene = EnergyFluid(nu=0.1, rho='rho', cp=2.0, kappa=5, dim=2,  time=False, visc_heating=False)
    >>> ene.pprint()
      temperauture_fluid: 2.0*(u(x, y)*Derivative(T(x, y), x) + v(x, y)*Derivative(T(x, y), y))*rho(x, y) - u(x, y)*Derivative(p(x, y), x) - v(x, y)*Derivative(p(x, y), y) - 5*Derivative(T(x, y), (x, 2)) - 5*Derivative(T(x, y), (y, 2))
    """

    def __init__(
        self,
        cp="cp",
        kappa="kappa",
        rho="rho",
        nu="nu",
        visc_heating=False,
        dim=3,
        time=False,
    ):
        # set params
        self.dim = dim
        self.time = time
        self.nu = nu
        self.rho = rho
        self.visc_heating = visc_heating

        # specific heat
        self.cp = cp

        # conductivity
        self.kappa = kappa

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t = Symbol("t")

        # velocity componets
        u = Function("u")(x, y, z, t)
        v = Function("v")(x, y, z, t)
        w = Function("w")(x, y, z, t)

        # Density
        rho = Function("rho")(x, y, z, t)

        # kinematic viscosity
        nu = Function("nu")(x, y, z, y)

        # pressure
        p = Function("p")(x, y, z, t)

        # Temperature
        T = Function("T")(x, y, z, t)

        # viscous heat dissipation
        vel = [u, v, w]
        coord = [x, y, z]
        visc_h = 0 * x
        if visc_heating == True:
            for i, j in zip(range(0, 3), range(0, 3)):
                visc_h = visc_h + (
                    vel[i].diff(coord[j]) * vel[i].diff(coord[j])
                    + vel[i].diff(coord[j]) * vel[j].diff(coord[i])
                    - 2 / 3 * vel[i].diff(coord[i]) * vel[j].diff(coord[j])
                )

        visc_h = nu * rho * visc_h

        # pressure work
        p_work = (
            0 * x
            if type(self.rho) == float
            else (p.diff(t) + u * (p.diff(x)) + v * (p.diff(y)) + w * (p.diff(z)))
        )

        # set equations
        self.equations = {}
        self.equations["temperauture_fluid"] = (
            rho * cp * (T.diff(t) + u * (T.diff(x)) + v * (T.diff(y)) + w * (T.diff(z)))
            - kappa * (T.diff(x)).diff(x)
            - kappa * (T.diff(y)).diff(y)
            - kappa * (T.diff(z)).diff(z)
            - p_work
            - visc_h
        )

        input_variables = {"x": x, "y": y, "z": z, "t": t}
        if self.dim == 2:
            input_variables.pop("z")
        if not self.time:
            input_variables.pop("t")
        self.subs(u, Function("u")(*input_variables))
        self.subs(v, Function("v")(*input_variables))
        self.subs(w, Function("w")(*input_variables))
        self.subs(T, Function("T")(*input_variables))
        self.subs(p, Function("p")(*input_variables))
        self.subs(nu, Function("nu")(*input_variables))
        self.subs(rho, Function("rho")(*input_variables))
        if type(self.rho) == float:
            self.subs(Function("rho")(*input_variables), self.rho)
        if type(self.nu) == float:
            self.subs(Function("nu")(*input_variables), self.nu)
