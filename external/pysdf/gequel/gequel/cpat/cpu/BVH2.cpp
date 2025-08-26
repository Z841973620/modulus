
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

#include "gequel/cpat/cpu/BVH2.h"

#include <stack>
#include <queue>

namespace gequel {
  /* find closest points */
  namespace cpat {
    namespace cpu {

      using gequel::core::BVH2;
        
      BVH2Engine::PointData::PointData(const std::vector<vec3f> &vertices, const std::vector<vec3i> &faces)
        : vertices(vertices), faces(faces)
      {
        std::vector<box3f> primBounds;
        for (auto face : faces) {
          primBounds.emplace_back(vertices[face[0]]);
          primBounds.back().extend(vertices[face[1]]);
          primBounds.back().extend(vertices[face[2]]);
        }
        bvh2.build(primBounds);
      }

      abstract::PointData *
      BVH2Engine::createModel(const std::vector<vec3f> &vertices,
                              const std::vector<vec3i> &faces) {
        return new BVH2Engine::PointData(vertices, faces);
      }

      void BVH2Engine::findCPATs(CPATResult *resultArray, int numQueries,
                                 /*! the point model of N points that we're
                                     querying from */
                                 abstract::PointData *_model,
                                 /*! the M positions for which we're querying
                                     the kNNs */
                                 const vec3f *queries,
                                 const float maxQueryDistance) {
        BVH2Engine::PointData *model = (BVH2Engine::PointData *)_model;
        owl::parallel_for(numQueries, [&](size_t queryID) {
          const vec3f queryPos = queries[queryID];
          CPATResult& result = resultArray[queryID];
          result.dist = maxQueryDistance;
          result.point = vec3f(0, 0, 0);
          result.triID = -1;
          doQuery(result, model, queryPos);
        });
      }

      void BVH2EngineBT::doQuery(CPATResult &result,
                                 const BVH2Engine::PointData *model,
                                 const vec3f &queryPos) {
        std::stack<std::pair<int,float>> travStack;
        travStack.push({0,0.f});
        
        while (!travStack.empty()) {
          auto top = travStack.top();
          travStack.pop();

          if (top.second > result.dist)
            continue;

          const int nodeID = top.first;
          const BVH2::Node &node = model->bvh2.nodes[nodeID];
          
          if (node.count == 0) {
            const BVH2::Node &n0 = model->bvh2.nodes[node.index+0];
            const BVH2::Node &n1 = model->bvh2.nodes[node.index+1];
            const float d0 = distance(queryPos,n0.bounds);
            const float d1 = distance(queryPos,n1.bounds);
            if (d0 < d1) {
              if (d1 <= result.dist)
                travStack.push({node.index+1,d1});
              if (d0 <= result.dist)
                travStack.push({node.index+0,d0});
            } else {
              if (d0 <= result.dist)
                travStack.push({node.index+0,d0});
              if (d1 <= result.dist)
                travStack.push({node.index+1,d1});
            }
            continue;
          }

          const uint32_t *leafList = model->bvh2.leafLists.data()+node.index;
          for (int i=0;i<node.count;i++) {
            const int   triID = leafList[i];
            const vec3i tri = model->faces[triID];
            vec3f closest;
            float nodeDist = pointTriDistance(queryPos, model->vertices[tri[0]],
                                              model->vertices[tri[1]],
                                              model->vertices[tri[2]], closest);
            if (nodeDist < result.dist) {
              result.triID = triID;
              result.point = closest;
              result.dist = nodeDist;
            }
          }
        }
      }

      void BVH2EnginePQ::doQuery(CPATResult &result,
                                 const BVH2Engine::PointData *model,
                                 const vec3f &queryPos) {
        std::priority_queue<std::pair<float,int>,
                            std::vector<std::pair<float,int>>,
                            std::greater<std::pair<float,int>>> pq;
        pq.push({0.f,0});
        while (!pq.empty()) {
          auto top = pq.top();
          pq.pop();

          if (top.first > result.dist)
            break;

          const int nodeID = top.second;
          const BVH2::Node &node = model->bvh2.nodes[nodeID];
          
          if (node.count == 0) {
            const BVH2::Node &n0 = model->bvh2.nodes[node.index+0];
            const BVH2::Node &n1 = model->bvh2.nodes[node.index+1];
            const float d0 = distance(queryPos,n0.bounds);
            const float d1 = distance(queryPos,n1.bounds);
            if (d1 <= result.dist)
              pq.push({d1,node.index+1});
            if (d0 <= result.dist)
              pq.push({d0,node.index+0});
            continue;
          }

          const uint32_t *leafList = model->bvh2.leafLists.data()+node.index;
          for (int i=0;i<node.count;i++) {
            const int   triID = leafList[i];
            const vec3i tri = model->faces[triID];
            vec3f closest;
            float nodeDist = pointTriDistance(queryPos, model->vertices[tri[0]],
                                              model->vertices[tri[1]],
                                              model->vertices[tri[2]], closest);
            if (nodeDist < result.dist) {
              result.triID = triID;
              result.point = closest;
              result.dist = nodeDist;
            }
          }
        }
      }


    } // ::gequel::knn::cpu
  } // ::gequel::knn
} // ::gequel
