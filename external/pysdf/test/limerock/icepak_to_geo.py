# import SimNet library
from sympy import Symbol, Eq, tanh, Max
import numpy as np
import itertools

from geometry import Geometry, Box, Channel

class LimeRock(object):
  def __init__(self):

    # scale STL
    self.scale = 5 / 0.3
    self.translate = (-0.055, 0, 0)

    # set fins
    self.nr_fins = 47
    self.fin_gap = 0.0018

    # make solids
    self.copper = []

    # parse file
    print("parsing stl file...")
    self._parse_file('./stl_files/limerock.stl')
    print("finished parsing")

    # geo
    heat_sink_bounds = (-0.7, 0.7)
    self.geo = self.channel - self.copper

  def solid_names(self):
    return list(self.solids.keys())

  def _parse_file(self, filename):
    # Read file
    reader = open(filename)
    sdf = 0
    while True:
      line = reader.readline()
      if 'solid' == line.split(' ')[0]:
        solid_name = line.split(' ')[-1].rstrip()
        bounds_lower, bounds_upper = self.read_solid(reader)
        if solid_name == 'opening.1':
          self.geo_bounds_lower = bounds_lower
        elif solid_name == 'fan.1':
          self.geo_bounds_upper = bounds_upper
        elif solid_name == 'FIN':
          fin = Box(bounds_lower, bounds_upper)
          fin.repeat(self.scale * self.fin_gap,
                     repeat_lower=(0,0,0),
                     repeat_higher=(self.nr_fins-1,0,0),
                     center=_center(bounds_lower, bounds_upper))
          self.copper.append(fin)
        else:
          self.copper.append(Box(bounds_lower, bounds_upper))
      else:
        break
    self.copper = Geometry(Max(*[geo.sdf for geo in self.copper]))
    self.channel = Box(self.geo_bounds_lower,
                       self.geo_bounds_upper)


  def read_solid(self, reader):
    # solid pieces
    faces = []
    while True:
      line = reader.readline()
      split_line = line.split(' ')
      if len(split_line) == 0:
        break
      elif 'endsolid' == split_line[0]:
        break
      elif 'facet' == split_line[0]:
        curve = {}
        # read outer loop line
        _ = reader.readline()
        # read 3 vertices
        a_0 = [float(x) for x in reader.readline().split(' ')[-3:]]
        a_1 = [float(x) for x in reader.readline().split(' ')[-3:]]
        a_2 = [float(x) for x in reader.readline().split(' ')[-3:]]
        faces.append([a_0, a_1, a_2])
        # read end loop/end facet
        _ = reader.readline()
        _ = reader.readline()
    faces = np.array(faces)
    bounds_lower = (np.min(faces[...,0]), np.min(faces[...,1]), np.min(faces[...,2])) # flip axis
    bounds_upper = (np.max(faces[...,0]), np.max(faces[...,1]), np.max(faces[...,2]))
    bounds_lower = tuple([self.scale*x + t for x, t in zip(bounds_lower, self.translate)])
    bounds_upper = tuple([self.scale*x + t for x, t in zip(bounds_upper, self.translate)])
    return bounds_lower, bounds_upper

def _center(bounds_lower, bounds_upper):
  center_x = bounds_lower[0] + (bounds_upper[0] - bounds_lower[0])/2
  center_y = bounds_lower[1] + (bounds_upper[1] - bounds_lower[1])/2
  center_z = bounds_lower[2] + (bounds_upper[2] - bounds_lower[2])/2
  return center_x, center_y, center_z
