#pragma once
/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! Compute signed distance field
// Input:
// * mesh_vertices: Grouped in 3 for each point, which are grouped in 3 for each
// triangle. Vertices explicitly duplicated, which means that the index array
// isn't needed.
// * input_points: Points to compute (signed) distance field. Grouped in 3 for each 3D position.
// Output:
// * points_signed_distance: The signed distance for each point.

#ifndef SDF_INTERFACE 
    #ifdef _WIN32
    #  define SDFEXPORT __declspec( dllexport )
    #  define SDFIMPORT __declspec( dllimport )
    #else
    #  define SDFEXPORT
    #  define SDFIMPORT
    #endif

    #ifdef _WIN32
    #ifdef pysdf_EXPORTS
    #define SDF_INTERFACE SDFEXPORT
    #else
    #define SDF_INTERFACE SDFIMPORT
    #endif
    #else
    #define SDF_INTERFACE
    #endif
#endif


/**
 * Function to compute the signed distance field (SDF) on a set of input points given a triangulated input geometry.

 * @param[in]  num_vertices                     The number of *vertices* in the triangulated mesh. (e.g., `mesh_vertices.size()/3`)
 * @param[in]  mesh_vertices                    Flat array of the vertices. Vertices consist of three consecutive values,
 *                                              and triangles consist of three consecutive vertices (9 values total).
 *                                              Vertices are duplicated.
 * @param[in]  num_points                       The number of SDF evaluation points. (e.g., `eval_points.size()/3`)
 * @param[in]  input_points                     Flat array of evaluation points, where the SDF will be evaluated.
 *                                              Each point consists of 3 consecutive values.
 * @param[out] points_signed_distance           The computed signed distance at each input point.
 * @param[out] closest_hit_point                The closest point on a triangle corresponding to the input point.
 */
#ifdef __cplusplus
extern "C"
#endif
SDF_INTERFACE void signedDistanceField(int num_vertices, double* mesh_vertices,
                                    int num_points, double* input_points,
                                    double* points_signed_distance,
                                    double* closest_hit_point);



/**
 * Function to compute the distance field (DF) on a set of input points given a triangulated input geometry.

 * @param[in]  num_vertices       The number of *vertices* in the triangulated mesh. (e.g., `mesh_vertices.size()/3`)
 * @param[in]  mesh_vertices      Flat array of the vertices. Vertices consist of three consecutive values,
 *                                and triangles consist of three consecutive vertices (9 values total).
 *                                Vertices are duplicated.
 * @param[in]  num_points         The number of DF evaluation points. (e.g., `eval_points.size()/3`)
 * @param[in]  input_points       Flat array of evaluation points, where the DF will be evaluated.
 *                                Each point consists of 3 consecutive values.
 * @param[out] points_distance    The computed distance at each input point.
 * @param[out] closest_hit_point  The closest point on a triangle corresponding to the input point.
 */
#ifdef __cplusplus
extern "C"
#endif
SDF_INTERFACE void distanceField(int num_vertices, double* mesh_vertices,
                             int num_points, double* input_points,
                             double* points_distance,
                             double* closest_hit_point);


/**
 * Function to compute the signed field (SF) on a set of input points given a triangulated input geometry.

 * @param[in]  num_vertices    The number of *vertices* in the triangulated mesh. (e.g., `mesh_vertices.size()/3`)
 * @param[in]  mesh_vertices   Flat array of the vertices. Vertices consist of three consecutive values,
 *                             and triangles consist of three consecutive vertices (9 values total).
 *                             Vertices are duplicated.
 * @param[in]  num_points      The number of SF evaluation points. (e.g., `eval_points.size()/3`)
 * @param[in]  input_points    Flat array of evaluation points, where the SF will be evaluated.
 *                             Each point consists of 3 consecutive values.
 * @param[out] inside_outside  The computed sign at each input point. Values >0 are outside object, <0 inside object.
 */
#ifdef __cplusplus
extern "C"
#endif
SDF_INTERFACE void insideOutside(int num_vertices, double* mesh_vertices,
                             int num_points, double* input_points,
                             double* inside_outside,
                             int nr_sphere_points);
