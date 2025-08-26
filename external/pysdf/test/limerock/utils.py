
import types
import numpy as np
from sympy import lambdify, Symbol
from pyevtk.hl import pointsToVTK

# functions to plot a variable
def var_to_vtk(var,
               save_name,
               coordinates=['x','y','z']):

  for axis in coordinates:
    if axis not in var.keys():
      var[axis] = np.zeros_like(next(iter(var.values())))
  pointsToVTK(save_name,
              var[coordinates[0]][:].copy(),
              var[coordinates[1]][:].copy(),
              var[coordinates[2]][:].copy(),
              data = {key: value[:].copy() for key, value in var.items()})

NP_LAMBDA_STORE = {}

def np_lambdify(f, r):
  """
  generates a numpy function from a sympy equation

  Parameters
  ----------
  f : Sympy Exp, float, int, bool
    the equation to convert to tensorflow.
    If float, int, or bool this gets converted
    to a constant function of value `f`.
  r : list, dict
    A list of the arguments for `f`. If dict then
    the keys of the dict are used.

  Returns
  -------
  np_f : numpy function
  """

  try:
    return NP_LAMBDA_STORE[(f, tuple(r.keys()))]
  except:
    pass
  try:
    if not isinstance(f, bool):
      f = float(f)
  except:
    pass
  if isinstance(f, (float, int)): # constant function
    def loop_lambda(constant):
      return lambda **x: np.zeros_like(next(iter(x.items()))[1]) + constant
    lambdify_f = loop_lambda(f)
  elif type(f) in [type((Symbol('x') > 0).subs(Symbol('x'), 1)), type((Symbol('x') > 0).subs(Symbol('x'), -1)), bool]: # TODO hacky sympy boolian check
    def loop_lambda(constant):
      if constant:
        return lambda **x: np.ones_like(next(iter(x.items()))[1], dtype=bool)
      else:
        return lambda **x: np.zeros_like(next(iter(x.items()))[1], dtype=bool)
    lambdify_f = loop_lambda(f)
  else:
    lambdify_f = lambdify([k for k in r], f, [NP_SYMPY_PRINTER, 'numpy'])
  NP_LAMBDA_STORE[(f, tuple(r))] = lambdify_f
  return lambdify_f

def _xor_np(x):
  return np.logical_xor(x)

def _min_np(x, axis=None):
  return_value = x[0]
  for value in x:
    return_value = np.minimum(return_value, value)
  return return_value

def _max_np(x, axis=None):
  return_value = x[0]
  for value in x:
    return_value = np.maximum(return_value, value)
  return return_value

def _heaviside_np(x):
  return np.heaviside(x, 0)

def _equal_np(x, y):
  return np.isclose(x, y)

NP_SYMPY_PRINTER = {'amin': _min_np,
                    'amax': _max_np,
                    'Heaviside': _heaviside_np,
                    'equal': _equal_np,
                    'Xor': _xor_np}
