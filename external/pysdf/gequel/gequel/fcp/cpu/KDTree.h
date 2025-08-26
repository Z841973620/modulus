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
#include "gequel/common/cpu/datastructures/kdtree.h"

namespace gequel {
  /* find closest points */
  namespace fcp {
    namespace cpu {

      struct KDTree : public FCPEngine {
        using Tree = core::KDTree;

        struct PointData : public abstract::PointData {
          PointData(const std::vector<vec3f> &points)
            : bounds(Tree::computeBounds(points)),
              nodes(Tree::buildTree(points,bounds))
          {}
          
          const box3f bounds;
          const std::vector<Tree::Node> nodes;
        };
    
        std::string toString() const override
        { return "fcp::cpu::KDTree"; }
        
        EngineType getType() const override { return GEQUEL_HOST; }
        abstract::PointData *createModel(const std::vector<vec3f> &points) override;
        void
        findClosestPoint(FCPResult *results, int numQueries,
                         abstract::PointData *model, const vec3f *queries,
                         const float maxQueryDistance =
                             std::numeric_limits<float>::infinity()) override;
        /*! the actual query kernel that traverses the kd-tree */
        FCPResult doQuery(const KDTree::PointData *model,
                          const vec3f &queryPos,
                          const float maxQueryDistance);
      };

    }
  }
}
