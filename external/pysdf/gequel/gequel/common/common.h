// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "owl/common/math/AffineSpace.h"
#include "owl/common/math/box.h"
#include "gequel/common/cpu/builtins.h"
// std
#include <vector>
#include <map>
#include <set>
#include <cstdio>

// cuda
#include <cuda_runtime.h>

#ifdef __CUDACC__
# define GQL_BOTH __device__ __host__
# define GQL_DEVICE __device__

#define CUDA_PTR_DEVICE(p)                                                     \
  {                                                                            \
    cudaPointerAttributes attribs;                                             \
    cudaPointerGetAttributes(&attribs, p);                                     \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "Pointer lookup (%s) failed with error code %d: %s\n",   \
              #p, err, cudaGetErrorString(err));                               \
      exit(3);                                                                 \
    }                                                                          \
    if (attribs.type != cudaMemoryTypeDevice) {                                \
      fprintf(stderr, "Invalid device pointer (%s) in %s at line %d\n", #p,    \
              __FUNCTION__, __LINE__);                                         \
      exit(3);                                                                 \
    }                                                                          \
  }

#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        exit( 2 );                                                      \
      }                                                                 \
  }

#else
# define GQL_BOTH /* nothing */
# define GQL_DEVICE /* nothing */
#define CUDA_PTR_DEVICE(p) /* nothing */
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#endif

namespace gequel {
  using namespace owl::common;

  inline GQL_BOTH float distance(const vec3f& point, const box3f& box)
  {
    vec3f closestPointInBox = min(max(point, box.lower), box.upper);
    return length(closestPointInBox - point);
  }

  inline GQL_BOTH box3f makeBox(float3 lower, float3 upper) {
    return box3f(*(vec3f*)&lower, *(vec3f*)&upper);
  }

  inline GQL_BOTH float pointTriDistance(const vec3f &point, const vec3f &a,
                                         const vec3f &b, const vec3f &c,
                                         vec3f &closest) {
    // Source: https://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    // Cast the problem as a quadratic program with linear constraints to find barycentric coordinates

    float bb = dot(b - a, b - a);
    float bc = dot(b - a, c - a);
    float cc = dot(c - a, c - a);
    float bp = dot(b - a, a - point);
    float cp = dot(c - a, a - point);
    //float pp = dot(a - point, a - point);

    float det = fabsf(bb * cc - bc * bc); // should always be positive, so negative value is numerical error

    // Compute global minimum of barycentric coordinates (before det division)
    float u =  bc * cp - cc * bp;
    float v = bc * bp - bb * cp;

    // Check barycentric constraints and find boundary point if appropriate
    int edge = 0;
    if (u + v <= det) {
      if (u < 0.f) {
        if (v < 0.f) {
          // region 4 - constrained min either on edge 2 (vertical) or 3 (horizontal)
          edge = bp < 0.f ? 3 : 2;
        } else {
          // region 3 - constrained min on edge 2 (vertical)
          edge = 2;
        }
      } else if (v < 0.f) {
        //region 5 - constrained min on edge 3 (horizontal)
        edge = 3;
      } else {
        // region 0 - global min in triangle
        //edge = 0;
      }
    } else {
      if (u < 0.f) {
        // region 2 - constrained min on edge 1 (diagonal) or 2 (vertical)
        float dirdivVertical = -cc - cp;
        edge = dirdivVertical < 0.f ? 2 : 1;
      } else if (v < 0.f) {
        // region 6 - constrained min on edge 1 (diagonal) or 3 (horizontal)
        float dirdivHorizontal = -bb - bp;
        edge = dirdivHorizontal < 0.f ? 3 : 1;
      } else {
        // region 1 - constrained min on edge 1 (diagonal)
        edge = 1;
      }
    }

    switch (edge) {
    case 0: // inside triangle
      u /= det;
      v /= det;
      break;
    case 1: {
      // edge u+v=1
      float numerator = (cc + cp) - (bc + bp);
      float denominator = bb - 2.f * bc + cc;
      if (numerator <= 0.f) {
        u = 0.f;
      } else if (numerator >= denominator) {
        u = 1.f;
      } else {
        u = numerator / denominator;
      }
      v = 1.f - u;
      break;
    }
    case 2: {
      // edge u=0
      u = 0.f;
      if (cp >= 0.f) {
        v = 0.f;
      } else if (-cp >= cc) {
        v = 1.f;
      } else {
        v = -cp / cc;
      }
      break;
    }
    case 3: {
      // edge v=0
      v = 0.f;
      if (bp >= 0.f) {
        u = 0.f;
      } else if (-bp >= bb) {
        u = 1.f;
      } else {
        u = -bp / bb;
      }
      break;
    }
    }

    closest = a + u * (b - a) + v * (c - a);
    return length(closest - point);
  }

} // namespace gequel
