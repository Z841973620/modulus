"""Wave equation
Reference: https://en.wikipedia.org/wiki/Wave_equation
"""

from sympy import Symbol, Function, Number

from modulus.eq.pde import PDE


class WaveEquation(PDE):
    """
    Wave equation

    Parameters
    ==========
    u : str
        The dependent variable.
    c : float, Sympy Symbol/Expr, str
        Wave speed coefficient. If `c` is a str then it is
        converted to Sympy Function of form 'c(x,y,z,t)'.
        If 'c' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    dim : int
        Dimension of the wave equation (1, 2, or 3). Default is 2.
    time : bool
        If time-dependent equations or not. Default is True.

    Examples
    ========
    >>> we = WaveEquation(c=0.8, dim=3)
    >>> we.pprint()
      wave_equation: u__t__t - 0.64*u__x__x - 0.64*u__y__y - 0.64*u__z__z
    >>> we = WaveEquation(c='c', dim=2, time=False)
    >>> we.pprint()
      wave_equation: -c**2*u__x__x - c**2*u__y__y - 2*c*c__x*u__x - 2*c*c__y*u__y
    """

    name = "WaveEquation"

    def __init__(self, u="u", c="c", dim=3, time=True):
        # set params
        self.u = u
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

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # wave speed coefficient
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)

        # set equations
        self.equations = {}
        self.equations["wave_equation"] = (
            u.diff(t, 2)
            - c ** 2 * u.diff(x, 2)
            - c ** 2 * u.diff(y, 2)
            - c ** 2 * u.diff(z, 2)
        )


class HelmholtzEquation(PDE):
    name = "HelmholtzEquation"

    def __init__(self, u, k, dim=3):
        """
        Helmholtz equation

        Parameters
        ==========
        u : str
            The dependent variable.
        k : float, Sympy Symbol/Expr, str
            Wave number. If `k` is a str then it is
            converted to Sympy Function of form 'k(x,y,z,t)'.
            If 'k' is a Sympy Symbol or Expression then this
            is substituted into the equation.
        dim : int
            Dimension of the wave equation (1, 2, or 3). Default is 2.
        """

        # set params
        self.u = u
        self.dim = dim

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # wave speed coefficient
        if type(k) is str:
            k = Function(k)(*input_variables)
        elif type(k) in [float, int]:
            k = Number(k)

        # set equations
        self.equations = {}
        self.equations["helmholtz"] = -(
            k ** 2 * u + u.diff(x, 2) + u.diff(y, 2) + u.diff(z, 2)
        )
