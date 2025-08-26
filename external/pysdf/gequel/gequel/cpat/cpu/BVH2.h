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

#include "gequel/cpat/cpat.h"
#include "gequel/common/cpu/datastructures/bvh2.h"

namespace gequel {
  /* find closest point on a triangle */
  namespace cpat {
    namespace cpu {

      struct BVH2Engine : public CPATEngine {

        struct PointData : public abstract::PointData {
          PointData(const std::vector<vec3f> &vertices, const std::vector<vec3i> &faces);

          core::BVH2 bvh2;
          const std::vector<vec3f> vertices;
          const std::vector<vec3i> faces;
        };
    
        EngineType getType() const override { return GEQUEL_HOST; }
        abstract::PointData *
        createModel(const std::vector<vec3f> &vertices,
                    const std::vector<vec3i> &faces) override;

        virtual void doQuery(CPATResult &result,
                             const BVH2Engine::PointData *model,
                             const vec3f &queryPos) = 0;
        void findCPATs(CPATResult *results, int numQueries,
                       abstract::PointData *model, const vec3f *queries,
                       const float maxQueryDistance =
                           std::numeric_limits<float>::infinity()) override;
      };

      /*! priority-queue based traversal */
      struct BVH2EnginePQ : public BVH2Engine {
        std::string toString() const override
        { return "cpat::cpu::BVH2::PQ"; }

        void doQuery(CPATResult &result, const BVH2Engine::PointData *model,
                     const vec3f &queryPos) override;
      };
      /*! stack-based backtracking traversal */
      struct BVH2EngineBT : public BVH2Engine {
        std::string toString() const override
        { return "cpat::cpu::BVH2::BT"; }
        
        void doQuery(CPATResult &result, const BVH2Engine::PointData *model,
                     const vec3f &queryPos) override;
      };
        
    }
  }
}
