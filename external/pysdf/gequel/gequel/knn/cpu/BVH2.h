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

#include "gequel/knn/knn.h"
#include "gequel/common/cpu/datastructures/bvh2.h"
#include "gequel/knn/CandidateSet.h"

namespace gequel {
  /* find closest points */
  namespace knn {
    namespace cpu {

      struct BVH2Engine : public KNNEngine {

        struct PointData : public abstract::PointData {
          PointData(const std::vector<vec3f> &points);

          core::BVH2 bvh2;
          const std::vector<vec3f> points;
        };
    
        EngineType getType() const override { return GEQUEL_HOST; }
        abstract::PointData *createModel(const std::vector<vec3f> &points) override;

        virtual void doQuery(InMemoryCandidateHeap &candidates,
                             const BVH2Engine::PointData *model,
                             const vec3f &queryPos) = 0;

        void findKNNs(/*! array of all k results for every of the
                          M queries. total array size must be k*M,
                          with i'the query result being returned
                          in elements starting at k*i. if a query
                          found less than k elements, then the
                          last entries in that query's k result
                          entries will have their pointID set to
                          -1 */
                      KNNResult *resultArray, int numQueries,
                      /*! the point model of N points that we're
                          querying from */
                      abstract::PointData *model,
                      /*! the M positions for which we're querying
                          the kNNs */
                      const vec3f *queries, const int k,
                      const float maxQueryDistance =
                          std::numeric_limits<float>::infinity()) override;
      };

      /*! priority-queue based traversal */
      struct BVH2EnginePQ : public BVH2Engine {
        std::string toString() const override
        { return "knn::cpu::BVH2::PQ"; }
        
        void doQuery(InMemoryCandidateHeap &candidates,
                     const BVH2Engine::PointData *model,
                     const vec3f &queryPos) override;
      };
      /*! stack-based backtracking traversal */
      struct BVH2EngineBT : public BVH2Engine {
        std::string toString() const override
        { return "knn::cpu::BVH2::BT"; }
        
        void doQuery(InMemoryCandidateHeap &candidates,
                     const BVH2Engine::PointData *model,
                     const vec3f &queryPos) override;
      };
        
    }
  }
}
