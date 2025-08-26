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

#include "gequel/fcp/cpu/BruteForce.h"

namespace gequel {
  /* find closest points */
  namespace fcp {
    namespace cpu {
      
      abstract::PointData *BruteForce::createModel(const std::vector<vec3f> &points)
      {
        if (points.size() > 10000)
          throw std::runtime_error("cowardly refusing to do brute-force queries in that many points .... ");
        return new BruteForce::PointData(points);
      }

      void BruteForce::findClosestPoint(FCPResult *results, int numQueries,
                                        abstract::PointData *_model,
                                        const vec3f *queries,
                                        const float maxQueryDistance) {
        assert(_model);
        BruteForce::PointData *model = (BruteForce::PointData *)_model;
        owl::parallel_for(numQueries, [&](size_t queryID) {
          const vec3f queryPoint = queries[queryID];
          FCPResult &result = results[queryID];
          result.pointID = -1;
          result.distance = maxQueryDistance;
          for (int pointID = 0; pointID < (int)model->points.size();
               pointID++) {
            float distance = length(model->points[pointID] - queryPoint);
            if (distance < result.distance) {
              result.distance = distance;
              result.pointID = pointID;
            }
          }
        });
      }
      } // namespace cpu
  }
}
