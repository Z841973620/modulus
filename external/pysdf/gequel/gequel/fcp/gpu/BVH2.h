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

#include "gequel/fcp/fcp.h"
#include "gequel/common/cpu/datastructures/bvh2.h"

#include <thrust/device_vector.h>

namespace gequel {
  /* find closest points */
  namespace fcp {
    namespace gpu {

      struct BVH2Engine : public FCPEngine {

        struct PointData : public abstract::PointData {
          explicit PointData(const std::vector<vec3f> &vertices);

          core::BVH2 bvh2;
          const thrust::device_vector<vec3f> d_points;
          thrust::device_vector<uint32_t> d_leafLists;
          thrust::device_vector<core::BVH2::Node> d_nodes;
        };
    
        EngineType getType() const override { return GEQUEL_GPU_NATIVE; }
        abstract::PointData *
        createModel(const std::vector<vec3f> &points) override;

        virtual void doQueryGPU(const core::BVH2::Node *nodes,
                                const uint32_t *leafLists,
                                const vec3f *points,
                                const vec3f *queryPos, int numQueries,
                                float maxQueryDistance,
                                FCPResult *results) = 0;

        void findClosestPoint(FCPResult *resultArray, int numQueries,
                              /*! the point model of N points that we're
                                  querying from */
                              abstract::PointData *model,
                              /*! the M positions for which we're querying
                                  the kNNs */
                              const vec3f *queries,
                              const float maxQueryDistance) override;
      };

      ///*! priority-queue based traversal */
      //struct BVH2EnginePQ : public BVH2Engine {
      //  std::string toString() const override
      //  { return "cpat::gpu::BVH2::PQ"; }
      //  
      //  void doQuery(InMemoryCandidateHeap &candidates,
      //               const BVH2Engine::PointData *model,
      //               const vec3f &queryPos) override;
      //};

      /*! stack-based backtracking traversal */
      struct BVH2EngineBT : public BVH2Engine {
        std::string toString() const override
        { return "fcp::gpu::BVH2::BT"; }

        void doQueryGPU(const core::BVH2::Node *nodes,
                        const uint32_t *leafLists, const vec3f *vertices,
                        const vec3f *queryPos, int numQueries,
                        float maxQueryDistance, FCPResult *results) override;
      };
    }
  }
}
