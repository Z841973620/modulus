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

#include <vector_types.h>

#include <vector>

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

enum SIGNED_FIELD_TECHNIQUE {
  SIGNED_FIELD_TECHNIQUE_INTERSECTION_COUNT = 0,
  SIGNED_FIELD_TECHNIQUE_NORMAL_ALIGNMENT
};

enum CELL_TYPE {
  CELL_TYPE_OUTSIDE = 0,
  CELL_TYPE_INSIDE
};

#ifdef __cplusplus
extern "C"
#endif
SDF_INTERFACE void computeDistanceField(const std::vector<float3> &mesh_vertices,
                          const std::vector<uint3>& mesh_indices,
                          const size_t num_query_pts, float3 *d_query_pts,
                          float *d_query_distance, float3 *d_query_hit_point,
                          int *d_query_triID);

#ifdef __cplusplus
extern "C"
#endif
SDF_INTERFACE void computeSignedDistanceField(const std::vector<float3>& mesh_vertices,
                                const std::vector<uint3>& mesh_indices,
                                const size_t num_query_pts,
                                float3* d_query_pts,
                                float* d_query_distance,
                                float3* d_query_hit_point,
                                int* d_query_triID,
                                const SIGNED_FIELD_TECHNIQUE signed_field_method = SIGNED_FIELD_TECHNIQUE_INTERSECTION_COUNT
                                );
