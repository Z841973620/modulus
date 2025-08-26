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

  
  /*! careful - unlike for fcp, each result will not just compute
      _one_ of those results, but always an array of 'k' of those */
  struct KNNResult {
    inline GQL_BOTH KNNResult() {}
    inline GQL_BOTH KNNResult(int pointID, float dist)
      : pointID(pointID), dist(dist)
    {}
    
    int   pointID;
    float dist;
  };

  struct KNNEngine {
    virtual std::string toString() const = 0;
    virtual EngineType getType() const = 0;
    virtual abstract::PointData *createModel(const std::vector<vec3f> &points) = 0;
    /*! finds, for each point in the query set, the closest point in
      the point data set */
    void findKNNs(/*! array of all k results for every of the
                      M queries. total array size must be k*M,
                      with i'the query result being returned
                      in elements starting at k*i. if a query
                      found less than k elements, then the
                      last entries in that query's k result
                      entries will have their pointID set to
                      -1 */
                  std::vector<KNNResult> &resultArray,
                  /*! the point model of N points that we're
                      querying from */
                  abstract::PointData *model,
                  /*! the M positions for which we're querying
                      the kNNs */
                  const std::vector<vec3f> &queries, const int k,
                  const float maxQueryDistance =
                      std::numeric_limits<float>::infinity()) {
      assert(resultArray.size() == queries.size() * k);
      assert(k > 0);
      assert(model);
      findKNNs(resultArray.data(), queries.size(), model, queries.data(), k,
               maxQueryDistance);
    }

    virtual void findKNNs(/*! array of all k results for every of the
                              M queries. total array size must be k*M,
                              with i'the query result being returned
                              in elements starting at k*i. if a query
                              found less than k elements, then the
                              last entries in that query's k result
                              entries will have their pointID set to
                              -1 */
                          KNNResult *resultArray,
                          int numQueries,
                          /*! the point model of N points that we're
                              querying from */
                          abstract::PointData *model,
                          /*! the M positions for which we're querying
                              the kNNs */
                          const vec3f *queries,
                          const int   k,
                          const float maxQueryDistance
                          = std::numeric_limits<float>::infinity()) = 0;
  };
  
}
