
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

#include "gequel/knn/cpu/BruteForce.h"

namespace gequel {
  /* find closest points */
  namespace knn {
    namespace cpu {
      
      abstract::PointData *BruteForce::createModel(const std::vector<vec3f> &points)
      {
        if (points.size() > 12000)
          throw std::runtime_error("cowardly refusing to do brute-force queries in that many points .... ");
        return new BruteForce::PointData(points);
      }

      void BruteForce::findKNNs(/*! array of all k results for every of the
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
                                abstract::PointData *_model,
                                /*! the M positions for which we're querying
                                    the kNNs */
                                const vec3f *queries, const int k,
                                const float maxQueryDistance)
          {
        BruteForce::PointData *model = (BruteForce::PointData *)_model;
        owl::parallel_for(numQueries, [&](size_t queryID) {
          const vec3f queryPos = queries[queryID];
          InMemoryCandidateHeap candidates(resultArray + queryID * k, k,
                                           maxQueryDistance);
          // PRINT(queryPos);
          for (int pointID = 0; pointID < model->points.size(); pointID++) {
            const vec3f point = model->points[pointID];
            // PRINT(point);
            const float dist = length(point - queryPos);
            // PRINT(dist);
            // PRINT(candidates.currentRadius());
            // PRINT(candidates.currentCount());
            maybe_insert(candidates, pointID, dist);
          }

          // finally, fill in unused k-nearest slots if we found
          // less than k entires
          for (int i = candidates.count; i < k; i++)
            candidates.memory[i] = {-1, maxQueryDistance};
        });
      }

    } // ::gequel::knn::cpu
  } // ::gequel::knn
} // ::gequel
