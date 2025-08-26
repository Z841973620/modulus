
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

#include "gequel/knn/cpu/BVH2.h"

#include <stack>
#include <queue>

namespace gequel {
  /* find closest points */
  namespace knn {
    namespace cpu {

      using gequel::core::BVH2;
        
      BVH2Engine::PointData::PointData(const std::vector<vec3f> &points)
        : points(points)
      {
        std::vector<box3f> primBounds;
        for (auto point : points)
          // iw - yes i know it's wasteful to store each point as a
          // box, but this way allows to reuse the bvh2 class and
          // builder also for triangles
          primBounds.push_back(box3f(point,point));

        bvh2.build(primBounds);
      }

      
      abstract::PointData *BVH2Engine::createModel(const std::vector<vec3f> &points)
      {
        return new BVH2Engine::PointData(points);
      }

      void BVH2Engine::findKNNs(/*! array of all k results for every of the
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
                                const float maxQueryDistance) {
        BVH2Engine::PointData *model = (BVH2Engine::PointData *)_model;
        owl::parallel_for(numQueries, [&](size_t queryID) {
          const vec3f queryPos = queries[queryID];
          InMemoryCandidateHeap candidates(resultArray + queryID * k, k,
                                           maxQueryDistance);
          doQuery(candidates, model, queryPos);
        });
      }

      void BVH2EngineBT::doQuery(InMemoryCandidateHeap &candidates,
                               const BVH2Engine::PointData *model,
                               const vec3f &queryPos)
      {
        std::stack<std::pair<int,float>> travStack;
        travStack.push({0,0.f});
        
        while (!travStack.empty()) {
          auto top = travStack.top();
          travStack.pop();

          const float currentR = candidates.currentRadius();
          if (top.second > currentR)
            continue;

          const int nodeID = top.first;
          const BVH2::Node &node = model->bvh2.nodes[nodeID];
          
          if (node.count == 0) {
            const BVH2::Node &n0 = model->bvh2.nodes[node.index+0];
            const BVH2::Node &n1 = model->bvh2.nodes[node.index+1];
            const float d0 = distance(queryPos,n0.bounds);
            const float d1 = distance(queryPos,n1.bounds);
            if (d0 < d1) {
              if (d1 <= currentR)
                travStack.push({node.index+1,d1});
              if (d0 <= currentR)
                travStack.push({node.index+0,d0});
            } else {
              if (d0 <= currentR)
                travStack.push({node.index+0,d0});
              if (d1 <= currentR)
                travStack.push({node.index+1,d1});
            }
            continue;
          }

          const uint32_t *leafList = model->bvh2.leafLists.data()+node.index;
          for (int i=0;i<node.count;i++) {
            const int   pointID = leafList[i];
            const vec3f point   = model->points[pointID];
            float nodeDist = length(point - queryPos);
            maybe_insert(candidates, pointID, nodeDist);
          }
        }
        // finally, fill in unused k-nearest slots if we found
        // less than k entires
        for (int i = candidates.count; i < candidates.k; i++)
          candidates.memory[i] = {-1, candidates.maxRadius};
      }


      void BVH2EnginePQ::doQuery(InMemoryCandidateHeap &candidates,
                               const BVH2Engine::PointData *model,
                               const vec3f &queryPos)
      {
        std::priority_queue<std::pair<float,int>,
                            std::vector<std::pair<float,int>>,
                            std::greater<std::pair<float,int>>> pq;
        pq.push({0.f,0});
        while (!pq.empty()) {
          auto top = pq.top();
          pq.pop();

          const float currentR = candidates.currentRadius();
          if (top.first > currentR)
            break;

          const int nodeID = top.second;
          const BVH2::Node &node = model->bvh2.nodes[nodeID];
          
          if (node.count == 0) {
            const BVH2::Node &n0 = model->bvh2.nodes[node.index+0];
            const BVH2::Node &n1 = model->bvh2.nodes[node.index+1];
            const float d0 = distance(queryPos,n0.bounds);
            const float d1 = distance(queryPos,n1.bounds);
            if (d1 <= currentR)
              pq.push({d1,node.index+1});
            if (d0 <= currentR)
              pq.push({d0,node.index+0});
            continue;
          }

          const uint32_t *leafList = model->bvh2.leafLists.data()+node.index;
          for (int i=0;i<node.count;i++) {
            const int   pointID = leafList[i];
            const vec3f point   = model->points[pointID];
            float nodeDist = length(point - queryPos);
            maybe_insert(candidates, pointID, nodeDist);
          }
        }
        
        // finally, fill in unused k-nearest slots if we found
        // less than k entires
        for (int i = candidates.count; i < candidates.k; i++)
          candidates.memory[i] = {-1, candidates.maxRadius};
      }


    } // ::gequel::knn::cpu
  } // ::gequel::knn
} // ::gequel
