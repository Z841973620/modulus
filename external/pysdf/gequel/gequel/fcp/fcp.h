
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

#include "../gequel.h"

namespace gequel {
  
  struct FCPResult {
    int   pointID;
    float distance;
  };

  typedef enum {
                GEQUEL_HOST,
                GEQUEL_GPU_NATIVE,
                GEQUEL_GPU_OFFLOAD
  } EngineType;

  namespace abstract {
    struct PointData {
      virtual ~PointData() {}
    };
  }
  
  struct FCPEngine {
    virtual std::string toString() const = 0;
    virtual EngineType getType() const = 0;
    virtual abstract::PointData *createModel(const std::vector<vec3f> &points) = 0;

    void findClosestPoint(
        std::vector<FCPResult> &results, abstract::PointData *model,
        const std::vector<vec3f> &queries,
        const float maxQueryDistance = std::numeric_limits<float>::infinity()) {
      assert(results.size() == queries.size());
      assert(model);
      findClosestPoint(results.data(), queries.size(), model, queries.data(),
                       maxQueryDistance);
    }

    /*! finds, for each point in the query set, the closest point in
        the point data set */
    virtual void findClosestPoint(FCPResult *results,
                                  int numQueries,
                                  abstract::PointData *model,
                                  const vec3f *queries,
                                  const float maxQueryDistance
                                  = std::numeric_limits<float>::infinity()) = 0;
  };
  
}
