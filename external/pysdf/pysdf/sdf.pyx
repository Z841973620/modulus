# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import numpy as np
cimport numpy as np

import ctypes


cdef extern from "sdf.h":

    void signedDistanceField(int num_vertices, double* mesh_vertices,
                             int num_points, double* input_points,
                             double* points_signed_distance,
                             double* closest_hit_point)

    void distanceField(int num_vertices, double* mesh_vertices,
                       int num_points, double* input_points,
                       double* points_signed_distance,
                       double* closest_hit_point)

    void insideOutside(int num_vertices, double* mesh_vertices,
                       int num_points, double* input_points,
                       double* inside_outside,
                       int nr_sphere_points)


def signed_distance_field(np.ndarray[double, ndim=1, mode="c"] mesh_vertices,
                          np.ndarray[double, ndim=1, mode="c"] input_points,
                          include_hit_points=False):
    """Input:
    mesh_vertices - 1D array of mesh triangles. Grouped in 3 for each 3d vertex in groups of three for each triangle.
    input_points - 1D array of points, grouped in 3 to describe each 3D point that needs a signed distance.
    include_hit_points - Boolean option to include the hit-and-optimized point used to compute the distance. Default=False

    Output:
    signed_distance_points - 1D array of distances, corresponding to the input_points.
                             Distances >0 are outside object, distances <0 are inside object.
    (optional) hit_points - 1D array of the ray-hit points used to compute the distance. Grouped in X,Y,Z coordinates groups.

    """
    cdef np.ndarray[double] signed_distance_points = np.zeros(int(len(input_points)/3))
    cdef np.ndarray[double] hit_points = np.zeros(int(len(input_points)))
    signedDistanceField(int(len(mesh_vertices)/3), &mesh_vertices[0],
                        int(len(input_points)/3), &input_points[0],
                        &signed_distance_points[0],
                        &hit_points[0])

    if not include_hit_points:
        return signed_distance_points
    else:
        return (signed_distance_points, hit_points)

def distance_field(np.ndarray[double, ndim=1, mode="c"] mesh_vertices,
                   np.ndarray[double, ndim=1, mode="c"] input_points,
                   include_hit_points=False):
    """Input:
    mesh_vertices - 1D array of mesh triangles. Grouped in 3 for each 3d vertex in groups of three for each triangle.
    input_points - 1D array of points, grouped in 3 to describe each 3D point that needs a signed distance.
    include_hit_points - Boolean option to include the hit-and-optimized point used to compute the distance. Default=False
    Output:
    distance_points - 1D array of distances, corresponding to the input_points.
                      Distances are always positive.
    (optional) hit_points - 1D array of the ray-hit points used to compute the distance. Grouped in X,Y,Z coordinates groups.
    
    """
    cdef np.ndarray[double] signed_distance_points = np.zeros(int(len(input_points)/3))
    cdef np.ndarray[double] hit_points = np.zeros(int(len(input_points)))
    distanceField(int(len(mesh_vertices)/3), &mesh_vertices[0],
                  int(len(input_points)/3), &input_points[0],
                  &signed_distance_points[0],
                  &hit_points[0])

    if not include_hit_points:
        return signed_distance_points
    else:
        return (signed_distance_points, hit_points)


def inside_outside(np.ndarray[double, ndim=1, mode="c"] mesh_vertices,
                   np.ndarray[double, ndim=1, mode="c"] input_points,
                   nr_sphere_points=101):
    """Input:
    mesh_vertices - 1D array of mesh triangles. Grouped in 3 for each 3d vertex in groups of three for each triangle.
    input_points - 1D array of points, grouped in 3 to describe each 3D point that needs a signed distance.
    nr_sphere_points - int, number or rays to use to check if inside or outside mesh.
    Output:
    inside_outside - 1D array of corresponding to if the point is inside or outside.
                     0 if outside and 1 if inside.
    """
    cdef np.ndarray[double] inside_outside = np.zeros(int(len(input_points)/3))
    insideOutside(int(len(mesh_vertices)/3), &mesh_vertices[0],
                  int(len(input_points)/3), &input_points[0],
                  &inside_outside[0],
                  nr_sphere_points)
    return inside_outside
