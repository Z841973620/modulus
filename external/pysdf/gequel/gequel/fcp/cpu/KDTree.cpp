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

#include "gequel/fcp/cpu/KDTree.h"

namespace gequel {
  /* find closest points */
  namespace fcp {
    namespace cpu {
      abstract::PointData *KDTree::createModel(const std::vector<vec3f> &points)
      {
        return new KDTree::PointData(points);
      }

      FCPResult KDTree::doQuery(const KDTree::PointData *model,
                                const vec3f &queryPos,
                                const float maxQueryDistance)
      {
        FCPResult result;
        result.distance = maxQueryDistance;
        result.pointID = -1;

        std::stack<std::pair<int,box3f>> travStack;
        travStack.push({0,model->bounds});

        while (!travStack.empty()) {
          auto top = travStack.top();
          travStack.pop();

          if (distance(queryPos,top.second) >= result.distance)
            continue;

          const int nodeID = top.first;
          const Tree::Node &node = model->nodes[nodeID];
          
          float nodeDist = length(node.pos-queryPos);
          if (nodeDist < result.distance) {
            result.distance = nodeDist;
            result.pointID  = node.pointID;
          }

          box3f lBounds = top.second;
          box3f rBounds = top.second;
          lBounds.upper[node.dim] = rBounds.lower[node.dim] = node.pos[node.dim];

          if (queryPos[node.dim] < node.pos[node.dim]) {
            // query pos is in left subtree - traverse left first
            // (which means pushing it _after_ the right side gets pushed)
            if ((2*nodeID+2) < model->nodes.size())
              travStack.push({2*nodeID+2,rBounds});
            if ((2*nodeID+1) < model->nodes.size()
                &&
                distance(queryPos,lBounds) < result.distance)
              travStack.push({2*nodeID+1,lBounds});
          } else {
            if ((2*nodeID+1) < model->nodes.size())
              travStack.push({2*nodeID+1,lBounds});
            if ((2*nodeID+2) < model->nodes.size()
                &&
                distance(queryPos,rBounds) < result.distance)
              travStack.push({2*nodeID+2,rBounds});
          }
        }
        return result;
      }

      void KDTree::findClosestPoint(FCPResult *results, int numQueries,
                                    abstract::PointData *_model,
                                    const vec3f *queries,
                                    const float maxQueryDistance) {
        assert(_model);
        KDTree::PointData *model = (KDTree::PointData *)_model;
        owl::parallel_for(numQueries, [&](size_t queryID) {
          results[queryID] = doQuery(model, queries[queryID], maxQueryDistance);
        });
      }
      } // namespace cpu
  }
}
