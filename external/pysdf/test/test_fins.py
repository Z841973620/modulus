# test

import pysdf.sdf as pysdf
import numpy as np
from stl import mesh as np_mesh
import matplotlib.pyplot as plt

def test_sdf():
    # read in mesh
    mesh = np_mesh.Mesh.from_file('fins.stl')

    # points to test on
    points_z = np.linspace(0.5, 0.6, 20000).astype(np.float64)
    points_x = np.full_like(points_z, 0.5)
    points_y = np.full_like(points_z, 0.5)
    points = np.stack([points_x, points_y, points_z], axis=1)
    points = points.flatten()

    # make sdf function
    triangles = np.array(mesh.vectors, dtype=np.float64).flatten()

    # make sdf points
    sdf_field, sdf_hit = pysdf.signed_distance_field(triangles,
                                                     points,
                                                     include_hit_points=True,
                                                     nr_sphere_points_distance_field=2000,
                                                     nr_sphere_points_inside_outside=2000)
    sdf_derivative = -(sdf_hit - points)
    sdf_derivative = np.reshape(sdf_derivative, (sdf_derivative.shape[0]//3, 3))
    sdf_derivative = sdf_derivative / np.linalg.norm(sdf_derivative, axis=1, keepdims=True)

    # make inside outside points
    inside_outside = pysdf.inside_outside(triangles,
                                          points,
                                          nr_sphere_points=2000)

    # plot sdf
    plt.plot(points_z, sdf_field) 
    plt.show()

    # plot inside outside
    plt.plot(points_z, inside_outside) 
    plt.show()

    # plot sdf derivative (this seems to be correct)
    plt.plot(points_z, sdf_derivative[:,2]) 
    plt.show()

    return (points_x, points_y, points_z), sdf_field

# test_sdf()


def test_broken_point():
    # read in mesh
    mesh = np_mesh.Mesh.from_file('fins.stl')

    # 2 points to test on
    # 1st correctly inside, 2nd incorrectly outside
    points = np.array([0.5, 0.5, 0.5001200060003, 0.5, 0.5, 0.5001250062503125])

    # make sdf function
    triangles = np.array(mesh.vectors, dtype=np.float64).flatten()

    # make sdf points
    sdf_field, sdf_hit = pysdf.signed_distance_field(triangles,
                                                     points,
                                                     include_hit_points=True)

    print(sdf_field)


if __name__ == "__main__":
    test_broken_point()
