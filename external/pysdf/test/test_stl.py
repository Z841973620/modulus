# test

import pysdf.sdf as pysdf
import numpy as np
from stl import mesh as np_mesh
import matplotlib.pyplot as plt

import argparse

import sys

from pyvtk import StructuredGrid, VtkData, Scalars

import IPython

def test_sdf(filename, N=100):
    # read in mesh
    try:
        mesh = np_mesh.Mesh.from_file(filename)
    except:
        print("Error opening {}".format(args.filename))        
        sys.exit(1)
    # points to test on
    NX = N
    NY = NX
    NZ = NX
    # cabin...
    # X, Y, Z = np.meshgrid(np.linspace(0.25, 2.25, NX),
    #                       np.linspace(-1.0, 1.0, NY),
    #                       np.linspace(0.0, 1.25, NZ))

    halo = 0.1
    mesh_min = mesh.vectors.min(axis=0).min(axis=0)
    mesh_max = mesh.vectors.max(axis=0).max(axis=0)

    xyz_min = [mesh_min[i] - abs(mesh_min[i])*halo for i in range(3)]
    xyz_max = [mesh_max[i] + abs(mesh_max[i])*halo for i in range(3)]

    X, Y, Z = np.meshgrid(np.linspace(xyz_min[0], xyz_max[0], NX),
                          np.linspace(xyz_min[1], xyz_max[1], NY),
                          np.linspace(xyz_min[2], xyz_max[2], NZ))

    # points_z = np.linspace(0.5, 0.6, 20000).astype(np.float64)
    # points_x = np.full_like(points_z, 0.5)
    # points_y = np.full_like(points_z, 0.5)
    points_xyz = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    points = points_xyz.flatten()
    
    # make sdf function
    triangles = np.array(mesh.vectors, dtype=np.float64).flatten()

    # make sdf points
    sdf_field, sdf_hit = pysdf.signed_distance_field(triangles,
                                                     points,
                                                     include_hit_points=True)
    sdf_derivative = -(sdf_hit - points)
    sdf_derivative = np.reshape(sdf_derivative, (sdf_derivative.shape[0]//3, 3))
    sdf_derivative = sdf_derivative / np.linalg.norm(sdf_derivative, axis=1, keepdims=True)

    points_vtk = []
    for i in range(points_xyz.shape[0]):
        points_vtk.append((points_xyz[i,0], points_xyz[i,1], points_xyz[i,2]))

    vtk = VtkData(StructuredGrid([NX, NY, NZ], points_vtk))
    vtk.point_data.append(Scalars(sdf_field, name="sdf"))
    vtk.tofile("saveme")
    # plot sdf
    # plt.plot(points_z, sdf_field) 
    # plt.show()

    # # plot inside outside
    # plt.plot(points_z, inside_outside) 
    # plt.show()

    # # plot sdf derivative (this seems to be correct)
    # plt.plot(points_z, sdf_derivative[:,2]) 
    # plt.show()

    return (points), sdf_field

# test_sdf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Display SDF for STL File")
    parser.add_argument('N', type=int, help="Number of SDF points")
    parser.add_argument('filename', type=str, help="stl file to load")
    
    
    args = parser.parse_args()
    print("Opening {}".format(args.filename))

    test_sdf(args.filename, args.N)

