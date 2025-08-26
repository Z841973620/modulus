
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

#include "gequel/knn/cpu/KDTree.h"

#include <stack>

namespace gequel {
  /* find closest points */
  namespace knn {
    namespace cpu {
      
      abstract::PointData *KDTree::createModel(const std::vector<vec3f> &points)
      {
        return new KDTree::PointData(points);
      }

      void KDTree::doQuery(InMemoryCandidateHeap &candidates,
                           const KDTree::PointData *model, vec3f queryPos) {
        std::stack<std::pair<int, box3f>> travStack;
        travStack.push({0, model->bounds});

        //PRINT(queryPos);
        while (!travStack.empty()) {
          auto top = travStack.top();
          travStack.pop();

          if (distance(queryPos, top.second) >= candidates.currentRadius())
            continue;

          const int nodeID = top.first;
          const Tree::Node &node = model->nodes[nodeID];

          //PRINT(node.pos);
          float nodeDist = length(node.pos - queryPos);
          //PRINT(nodeDist);
          //PRINT(candidates.currentRadius());
          //PRINT(candidates.currentCount());
          maybe_insert(candidates, node.pointID, nodeDist);

          box3f lBounds = top.second;
          box3f rBounds = top.second;
          lBounds.upper[node.dim] = rBounds.lower[node.dim] =
              node.pos[node.dim];

          if (queryPos[node.dim] < node.pos[node.dim]) {
            // query pos is in left subtree - traverse left first
            // (which means pushing it _after_ the right side gets pushed)
            if ((2 * nodeID + 2) < model->nodes.size())
              travStack.push({2 * nodeID + 2, rBounds});
            if ((2 * nodeID + 1) < model->nodes.size())
              travStack.push({2 * nodeID + 1, lBounds});
          } else {
            if ((2 * nodeID + 1) < model->nodes.size())
              travStack.push({2 * nodeID + 1, lBounds});
            if ((2 * nodeID + 2) < model->nodes.size())
              travStack.push({2 * nodeID + 2, rBounds});
          }
        }

        // finally, fill in unused k-nearest slots if we found
        // less than k entires
        for (int i = candidates.count; i < candidates.k; i++)
          candidates.memory[i] = {-1, candidates.maxRadius};
      }

      void KDTree::findKNNs(/*! array of all k results for every of the
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
        KDTree::PointData *model = (KDTree::PointData *)_model;
        owl::parallel_for(numQueries, [&](size_t queryID) {
          const vec3f queryPos = queries[queryID];
          InMemoryCandidateHeap candidates(resultArray + queryID * k, k,
                                           maxQueryDistance);
          doQuery(candidates, model, queryPos);
        });
      }

    } // ::gequel::knn::cpu
  } // ::gequel::knn
} // ::gequel
