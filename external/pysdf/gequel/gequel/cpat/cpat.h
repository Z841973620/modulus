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
#include "../common/common.h"
#include "../fcp/fcp.h"

namespace gequel {

  
  struct CPATResult {
    inline GQL_BOTH CPATResult() {}
    inline GQL_BOTH CPATResult(int triID, vec3f point, float dist)
      : triID(triID), point(point), dist(dist)
    {}
    
    int   triID;
    vec3f point;
    float dist;
  };

  struct CPATEngine {
    virtual std::string toString() const = 0;
    virtual EngineType getType() const = 0;
    virtual abstract::PointData *
    createModel(const std::vector<vec3f> &vertices,
                const std::vector<vec3i> &faces) = 0;
    /*! finds, for each point in the query set, the closest point in
      the triangle mesh */
    void findCPATs(/*! array of results for every of the
                       M queries. */
                  std::vector<CPATResult> &results,
                  /*! the point model of N points that we're
                      querying from */
                  abstract::PointData *model,
                  /*! the M positions for which we're querying
                      the kNNs */
                  const std::vector<vec3f> &queries,
                  const float maxQueryDistance =
                      std::numeric_limits<float>::infinity()) {
      assert(results.size() == queries.size());
      assert(model);
      findCPATs(results.data(), queries.size(), model, queries.data(),
                maxQueryDistance);
    }

    virtual void findCPATs(/*! array of results for every of the
                               M queries. */
                          CPATResult *results,
                          int numQueries,
                          /*! the point model of N points that we're
                              querying from */
                          abstract::PointData *model,
                          /*! the M positions for which we're querying
                              the kNNs */
                          const vec3f *queries,
                          const float maxQueryDistance
                          = std::numeric_limits<float>::infinity()) = 0;
  };
  
}
