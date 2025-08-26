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

namespace gequel {
  /* find closest points */
  namespace fcp {
    namespace cpu {

      struct BruteForce : public FCPEngine {
        struct PointData : public abstract::PointData {
          PointData(const std::vector<vec3f> &points)
            : points(points)
          {}
          const std::vector<vec3f> points;
        };
    
        std::string toString() const override
        { return "fcp::cpu::BruteForce"; }
        
        EngineType getType() const override { return GEQUEL_HOST; }
        abstract::PointData *createModel(const std::vector<vec3f> &points) override;
        void findClosestPoint(FCPResult *results,
                              int numQueries,
                              abstract::PointData *model,
                              const vec3f *queries,
                              const float maxQueryDistance
                              = std::numeric_limits<float>::infinity())  override;
      };

    }
  }
}
