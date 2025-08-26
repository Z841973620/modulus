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

#include <limits.h>
#include <float.h>

#include <optix.h>
#include <cuda_runtime.h>
#include <shared_types.h>

extern "C" __constant__ Params params;

static __forceinline__ __device__ void setPayloadF_0(float p) {
  optixSetPayload_0(__float_as_uint(p));
}

static __forceinline__ __device__ void setPayloadF_2(float p) {
  optixSetPayload_2(__float_as_uint(p));
}

static __forceinline__ __device__ float getPayloadF_0() {
  unsigned int val = optixGetPayload_0();
  return __uint_as_float(val);
}


__device__ __forceinline__ float3 getTriangleNormal(float3* vertexBuf, const unsigned int triId) {
  float3 v1 = vertexBuf[3*triId + 0];
  float3 v2 = vertexBuf[3*triId + 1];
  float3 v3 = vertexBuf[3*triId + 2];
  float3 normal = cross((v2 - v1), (v3 - v1));
  return normal;
}

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// arbitrary point signed field with counting to test
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

enum SIGNED_FIELD_HIT_STATUS {
  SIGNED_FIELD_HIT = 0,
  SIGNED_FIELD_DUPLICATE,
  SIGNED_FIELD_MISS
};


extern "C" __global__ void __raygen__signed_field_intersection() {

  const uint3 launch_index = optixGetLaunchIndex();
  int num_inside=0;
  int num_outside=0;
  for(int i=0; i<params.num_sphere_pts; i++) {
    
    // payload
    // 1. number of hits
    // 2. unused
    // 3. unused
    unsigned int payload0 = 0;
    unsigned int payload1 = 0;
    unsigned int payload2 = 0;
    
    float tmin = 0.0f;
    float tmax = 1000.0f;
    float ray_time = 0.0f;
    OptixVisibilityMask visibilityMask = 255;
    unsigned int rayFlags = OPTIX_RAY_FLAG_NONE;
    unsigned int SBToffset = 0;
    unsigned int SBTstride = 0;
    unsigned int missSBTIndex = 0;

    float3 ray_direction = normalize(params.d_sphere_pts[i]);
    float3 ray_origin = params.signed_field_pts[launch_index.x];

    optixTrace(params.handle, ray_origin, ray_direction, 0.0, tmax, ray_time,
               visibilityMask, rayFlags, SBToffset, SBTstride, missSBTIndex,
               payload0, payload1, payload2);

    if(payload0 == 0) {
      num_outside++;
    }
    else if(payload0 % 2 == 0) { // even
      num_outside++;
    }
    else { // odd
      num_inside++;
    }

  }

  if(num_inside > num_outside) {
    params.signed_field_results[launch_index.x] = CELL_TYPE_INSIDE;
  }
  else {
    params.signed_field_results[launch_index.x] = CELL_TYPE_OUTSIDE;
  }
}


extern "C" __global__ void __anyhit__signed_field() {
  // payload
  // 0. HIT/DUPLICATE/MISS
  // 1. number of hits
  // 2. unused

  unsigned int numhits = optixGetPayload_0();
  numhits += 1;
  optixSetPayload_0(numhits);
  optixIgnoreIntersection();
}

extern "C" __global__ void __miss__signed_field() {
  // empty
}
