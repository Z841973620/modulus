# import SimNet library
from sympy import Symbol, Eq, tanh, Abs
import sympy
import numpy as np
import pysdf.sdf as pysdf
from stl import mesh as np_mesh

from geometry import Box, Channel
from utils import var_to_vtk, np_lambdify

#########################################
# Construct Geometry With Geometry Module
#########################################
# geometry params for domain
channel_origin        = (-2.5, -0.5, -0.5625)
channel_dim           = (5.0, 1.0, 1.125)
heat_sink_base_origin = (-0.75, -0.6, -0.4375)
heat_sink_base_dim    = (0.65, 0.15, 0.875)
fin_origin            = (-0.75, -0.5, -0.4375)
fin_dim               = (0.65, 0.8625, 0.0075)
total_fins            = 17

# define sympy varaibles to parametize domain curves
x, y, z = Symbol('x'), Symbol('y'), Symbol('z')

# define geometry
# channel
channel = Channel(channel_origin, (channel_origin[0]+channel_dim[0],
                                   channel_origin[1]+channel_dim[1],
                                   channel_origin[2]+channel_dim[2]))

# fpga heat sink
heat_sink_base = Box(heat_sink_base_origin,
                     (heat_sink_base_origin[0]+heat_sink_base_dim[0], # base of heat sink
                      heat_sink_base_origin[1]+heat_sink_base_dim[1],
                      heat_sink_base_origin[2]+heat_sink_base_dim[2]))
fin_center = (fin_origin[0] + fin_dim[0]/2, fin_origin[1] + fin_dim[1]/2, fin_origin[2] + fin_dim[2]/2)
fin = Box(fin_origin, (fin_origin[0]+fin_dim[0],
                       fin_origin[1]+fin_dim[1],
                       fin_origin[2]+fin_dim[2]))
gap = (heat_sink_base_dim[2]-fin_dim[2])/(total_fins-1) # gap between fins
fin.repeat(gap, repeat_lower=(0,0,0), repeat_higher=(0,0,total_fins-1), center=fin_center)
fpga = heat_sink_base + fin

# entire geometry
geo = channel - fpga

##################################
# Construct Geometry from STL file
##################################
# make numpy stl mesh
mesh = np_mesh.Mesh.from_file("fpga_solid.stl")

# make sdf function
triangles = np.array(mesh.vectors, dtype=np.float64).flatten()

##########################################################
# Compute SDF for points from STL file and geometry module
##########################################################
# points to test on
# nr_points = 10000000
nr_points = 1000000
# nr_points = 10
np.random.seed(nr_points)
points_x = np.random.uniform(-2.5, 2.5, size=(nr_points)).astype(np.float64)
points_y = np.random.uniform(-0.5, 0.5, size=(nr_points)).astype(np.float64)
points_z = np.random.uniform(-0.5625, 0.5625, size=(nr_points)).astype(np.float64)
points = np.stack([points_x, points_y, points_z], axis=1)
points = points.flatten()

# Compute SDF for geometry module
geo_sdf_function = np_lambdify(geo.sdf, ['x', 'y', 'z'])
# geo_sdf_function2 = sympy.utilities.lambdify([x, y, z], geo.sdf)
# geo_sdf2 = geo_sdf_function2(x=points_x, y=points_y, z=points_z)
geo_sdf = geo_sdf_function(x=points_x, y=points_y, z=points_z)


# Compute SDF for pysdf module
sdf_field, sdf_hit = pysdf.distance_field(triangles,
                                          points,
                                          include_hit_points=True,
                                          nr_sphere_points=500, algorithm=0)
# sdf_field = -sdf_field # make inside outside for comparision

output_vtk = True

if output_vtk:
# make point cloud and save it for paraview
  point_cloud = {}
  point_cloud['x'] = points_x
  point_cloud['y'] = points_y
  point_cloud['z'] = points_z
  point_cloud['pysdf_sdf'] = sdf_field
  point_cloud['geometry_module_sdf'] = np.abs(geo_sdf)
  point_cloud['sdf_diff'] = np.abs(np.abs(geo_sdf) - sdf_field)
  var_to_vtk(point_cloud,
             "point_cloud")

  from pyevtk.hl import linesToVTK
  from pyevtk.hl import polyLinesToVTK
  import numpy as np
  x = np.zeros(2 * len(points)//3)
  y = np.zeros(2 * len(points)//3)
  z = np.zeros(2 * len(points)//3)
  xh = sdf_hit[0::3]
  yh = sdf_hit[1::3]
  zh = sdf_hit[2::3]

  x[0::2] = points_x
  x[1::2] = xh
  y[0::2] = points_y
  y[1::2] = yh
  z[0::2] = points_z
  z[1::2] = zh
    
  linesToVTK("sdf_hit", x, y, z, cellData={'sdf_diff': geo_sdf-sdf_field})
  polyLinesToVTK("sdf_hit2", x, y, z, 2*np.ones(x.size//2), cellData={'sdf_diff': np.abs(np.abs(geo_sdf)-sdf_field)})

np.testing.assert_allclose(np.abs(geo_sdf), sdf_field, atol=4e-2)

# # compute difference in points and save them
# positive_pysdf = np.greater(sdf_field, 0)
# positive_geo_sdf = np.greater(geo_sdf, 0)
# inside_pysdf_outside_geo_sdf = np.logical_and(positive_pysdf, np.logical_not(positive_geo_sdf))
# outside_pysdf_inside_geo_sdf = np.logical_and(positive_geo_sdf, np.logical_not(positive_pysdf))
# print("Total points: " + str(nr_points))
# num_wrong_inside = np.sum(inside_pysdf_outside_geo_sdf.astype(np.int))
# num_wrong_outside = np.sum(outside_pysdf_inside_geo_sdf.astype(np.int))
# print("Number of points inside pysdf but outside geometry module: " + str(num_wrong_inside))
# print("Number of points outside pysdf but inside geometry module: " + str(num_wrong_outside))
# if nr_points <= 1000000:
#   assert(num_wrong_inside/nr_points < 10e-06)
#   assert(num_wrong_outside/nr_points < 14e-06)
# elif nr_points <= 10000000:
#   assert(num_wrong_inside/nr_points < 100e-06)
#   assert(num_wrong_outside/nr_points < 100e-06)

# # save misclassifed points
# if output_vtk:
#   point_cloud_inside_pysdf_outside_geo_sdf = {}
#   for key, value in point_cloud.items():
#     point_cloud_inside_pysdf_outside_geo_sdf[key] = value[inside_pysdf_outside_geo_sdf]
#   var_to_vtk(point_cloud_inside_pysdf_outside_geo_sdf,
#              "point_cloud_inside_pysdf_outside_geo_sdf")
#   point_cloud_outside_pysdf_inside_geo_sdf = {}
#   for key, value in point_cloud.items():
#     point_cloud_outside_pysdf_inside_geo_sdf[key] = value[outside_pysdf_inside_geo_sdf]
#   var_to_vtk(point_cloud_outside_pysdf_inside_geo_sdf,
#              "point_cloud_outside_pysdf_inside_geo_sdf")





