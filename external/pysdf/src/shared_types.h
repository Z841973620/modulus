#pragma once
/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>

#include <optix.h>

#include <sdf_cpp.h>

#include "helper_math.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(A)                                     \
  do {                                                    \
    cudaError_t err = A;                                  \
    if (err != cudaSuccess) {                             \
      throw std::runtime_error(cudaGetErrorString(err));  \
    }                                                     \
  } while (false)

enum SDF_RAY_TYPES {
  SDF_RAY_TYPE_SURFACE_HIT = 0,
  SDF_RAY_TYPE_DEPTH_PROBE
};

#define NUM_ANYHIT 20

struct IntersectionPoints {
  float3 ray_origin;
  float3 hits[NUM_ANYHIT];
  unsigned int num_hits;
};

struct Mesh {
  std::vector<float3> vertices;
  float3* d_vertices;
  std::vector<uint3> triangles;
  uint3* d_triangles;
  float3 delta_xyz;
  float3 min;
};

struct Params {
  // grid projection parameters
  OptixTraversableHandle handle;

  // alternate signed field parameters
  float3* signed_field_pts;
  unsigned int* signed_field_results;
  SIGNED_FIELD_TECHNIQUE signed_technique;

  // distance field parameters
  unsigned int num_sphere_pts;
  unsigned int num_dist_field_pts;
  float3* d_sphere_pts;
  float3* distance_field_pts;

  // mesh info
  // note: Vertices are grouped in 3s by triangle with duplicated vertices.
  unsigned int num_mesh_points;
  float3* d_mesh_vertices;

  // distance output
  float* d_closest_hit_dist; // approx. closest distance from border cell to object surface
  float3* d_closest_hit_point;
};
