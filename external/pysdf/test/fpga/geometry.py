"""
Defines base class for all geometries
"""

import numpy as np
from sympy import Min, Symbol, Max, sin, cos, floor, Abs, sqrt
import copy
import itertools

class Geometry:
  """
  Constructive Geometry Module that allows sampling on surface and interior

  Parameters
  ----------
  sdf : SymPy Exprs
    SymPy expresion of signed distance function.
  """

  def __init__(self, sdf):
    self.sdf = sdf
    self.epsilon = 1e-5 # to check if in domain or outside

  def translate(self, xyz):
    """
    translate geometry

    Parameters
    ----------
    xyz : tuple of float, int or SymPy Symbol/Exp
      translate geometry by these values.
    """
    self.sdf = self.sdf.subs([(Symbol(dim), Symbol(dim)-x) for dim, x in zip(self.dims(), xyz)])
    self.boundary_criteria = None # reset boundary criteria

  def repeat(self, spacing, repeat_lower, repeat_higher, center=None):
    """
    finite repetition of geometry

    Parameters
    ----------
    spacing : float, int or SymPy Symbol/Exp
      spacing between each repetition.
    repeat_lower : tuple of ints
      How many repetitions going in negative direction.
    repeat_higher : tuple of ints
      How many repetitions going in positive direction.
    center : None, list of floats
      Do repetition with this center.
    """

    if center is not None:
      self.translate(tuple([-x for x in center]))
    assert len(repeat_lower) == len(self.dims())
    new_xyz = [Symbol(name+'_new') for name in self.dims()]
    clamped_xyz = [x-spacing*_clamp(_round(x/spacing), rl, rh) for x, rl, rh in zip(new_xyz, repeat_lower, repeat_higher)] # mod to repeat sdf
    self.sdf = self.sdf.subs([(Symbol(dim), clamped_x) for dim, clamped_x in zip(self.dims(), clamped_xyz)])
    self.sdf = self.sdf.subs([(Symbol(dim+'_new'), Symbol(dim)) for dim in self.dims()])
    if center is not None:
      self.translate(center)
    self.boundary_criteria = None # reset boundary criteria

  def dims(self):
    return ['x', 'y', 'z']

  def copy(self):
    return copy.deepcopy(self)

  def __add__(self, other):
    return Geometry(Max(self.sdf, other.sdf))

  def __sub__(self, other):
    return Geometry(Min(self.sdf, -other.sdf))

  def __invert__(self):
    return Geometry(-self.sdf)

  def __and__(self, other):
    return Geometry(Min(self.sdf, other.sdf))

class Channel(Geometry):
  """
  3D Channel (no bounding surfaces in x-direction)

  Parameters
  ==========
  point_1 : tuple with 3 ints or floats
    lower bound point of channel
  point_2 : tuple with 3 ints or floats
    upper bound point of channel
  """

  def __init__(self, point_1, point_2):
    # make sympy symbols to use
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    s_1, s_2 = Symbol('s_1'), Symbol('s_2')
    center = (point_1[0] + (point_2[0]-point_1[0])/2,
              point_1[1] + (point_2[1]-point_1[1])/2,
              point_1[2] + (point_2[2]-point_1[2])/2)
    side_x = point_2[0]-point_1[0]
    side_y = point_2[1]-point_1[1]
    side_z = point_2[2]-point_1[2]

    # calculate SDF
    y_dist = Abs(y - center[1]) - 0.5*side_y
    z_dist = Abs(z - center[2]) - 0.5*side_z
    outside_distance = sqrt(Max(y_dist, 0)**2 + Max(z_dist, 0)**2)
    inside_distance = Min(Max(y_dist, z_dist), 0)
    sdf = - (outside_distance + inside_distance)

    # initialize Channel
    super(Channel, self).__init__(sdf)

class Box(Geometry):
  """
  3D Box/Cuboid

  Parameters
  ==========
  point_1 : tuple with 3 ints or floats
    lower bound point of box
  point_2 : tuple with 3 ints or floats
    upper bound point of box
  """

  def __init__(self, point_1, point_2):
    # make sympy symbols to use
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    s_1, s_2 = Symbol('s_1'), Symbol('s_2')
    center = (point_1[0] + (point_2[0]-point_1[0])/2,
              point_1[1] + (point_2[1]-point_1[1])/2,
              point_1[2] + (point_2[2]-point_1[2])/2)
    side_x = point_2[0]-point_1[0]
    side_y = point_2[1]-point_1[1]
    side_z = point_2[2]-point_1[2]

    # calculate SDF
    x_dist = Abs(x - center[0]) - 0.5*side_x
    y_dist = Abs(y - center[1]) - 0.5*side_y
    z_dist = Abs(z - center[2]) - 0.5*side_z
    outside_distance = sqrt(Max(x_dist, 0)**2+Max(y_dist, 0)**2+Max(z_dist, 0)**2)
    inside_distance = Min(Max(x_dist, y_dist, z_dist), 0)
    sdf = - (outside_distance + inside_distance)

    # initialize Box
    super(Box, self).__init__(sdf)




def _clamp(x, lower, upper):
  return Min(Max(x, lower), upper)

def _round(x):
  return floor(x+1/2)

def _concat_numpy_dict_list(numpy_dict_list):
  concat_variable = {}
  for key in numpy_dict_list[0].keys():
    concat_variable[key] = np.concatenate([x[key] for x in numpy_dict_list], axis=0)
  return concat_variable

