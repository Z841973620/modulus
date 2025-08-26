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

#include "gequel/cpat/gpu/BVH2.h"

#include "gequel/common/gpu/datastructures/stack.h"

#include <stack>
#include <queue>
#include <device_launch_parameters.h>

namespace gequel {
  /* find closest points */
  namespace cpat {
    namespace gpu {

      using gequel::core::BVH2;
        
      BVH2Engine::PointData::PointData(const std::vector<vec3f> &vertices, const std::vector<vec3i> &faces)
        : d_vertices(vertices.begin(), vertices.end()), d_faces(faces.begin(), faces.end())
      {
        std::vector<box3f> primBounds;
        for (auto face : faces) {
          primBounds.emplace_back(vertices[face[0]]);
          primBounds.back().extend(vertices[face[1]]);
          primBounds.back().extend(vertices[face[2]]);
        }
        bvh2.build(primBounds);

        d_leafLists.assign(bvh2.leafLists.begin(), bvh2.leafLists.end());
        d_nodes.assign(bvh2.nodes.begin(), bvh2.nodes.end());
      }

      
      abstract::PointData *BVH2Engine::createModel(const std::vector<vec3f> &vertices, const std::vector<vec3i> &faces)
      {
        return new BVH2Engine::PointData(vertices, faces);
      }

      // Assume the arrays are already in device memory
      void BVH2Engine::findCPATs(CPATResult *resultArray, int numQueries,
                                /*! the point model of N points that we're
                                    querying from */
                                abstract::PointData *_model,
                                /*! the M positions for which we're querying
                                    the kNNs */
                                const vec3f *queries,
                                const float maxQueryDistance) {
        CUDA_PTR_DEVICE(resultArray);
        CUDA_PTR_DEVICE(queries);

        BVH2Engine::PointData *model = (BVH2Engine::PointData *)_model;
        doQueryGPU(thrust::raw_pointer_cast(&model->d_nodes[0]),
                   thrust::raw_pointer_cast(&model->d_leafLists[0]),
                   thrust::raw_pointer_cast(&model->d_vertices[0]),
                   thrust::raw_pointer_cast(&model->d_faces[0]), queries,
                   numQueries, maxQueryDistance, resultArray);
      }

      struct BVHCPATStackItem {
        float distance;
        int nodeID;
      };

      enum TraversalPath {
        FROM_PARENT,
        FROM_SIBLING,
        FROM_CHILD,
      };

      inline __device__ size_t siblingNode(const core::BVH2::Node* nodes, size_t parent, size_t child) {
        return nodes[parent].index == child ? child + 1 : child - 1;
      }

      __global__ void BVH2BTQuery(const core::BVH2::Node *nodes,
                                  const uint32_t *leafLists,
                                  const vec3f *vertices,
                                  const vec3i *faces,
                                  const vec3f *queryPos,
                                  int numQueries, float maxQueryDistance,
                                  CPATResult *results) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= numQueries)
          return;
        
        results[tid].dist = maxQueryDistance;
        results[tid].point = vec3f(0, 0, 0);
        results[tid].triID = -1;
        TraversalPath path = TraversalPath::FROM_PARENT;
        BVHCPATStackItem curItem;
        curItem.nodeID = 0;
        curItem.distance = 0.f;

        while (curItem.nodeID >= 0) {
          if (path == TraversalPath::FROM_CHILD && curItem.nodeID == 0) {
            break;
          }
          const BVH2::Node &node = nodes[curItem.nodeID];

          // TODO: Add fast path here in case node is too far and was taken off a
          // stack

          bool finishIter = false;
          if (node.count == 0) {
            const BVH2::Node &n0 = nodes[node.index + 0];
            const BVH2::Node &n1 = nodes[node.index + 1];

            BVHCPATStackItem leftItem, rightItem;
            leftItem.nodeID = node.index + 0;
            leftItem.distance = distance(queryPos[tid], n0.bounds);
            rightItem.nodeID = node.index + 1;
            rightItem.distance = distance(queryPos[tid], n1.bounds);
            bool onLeft = leftItem.distance < rightItem.distance;
            BVHCPATStackItem& nearNode = onLeft ? leftItem : rightItem;

            if (path != TraversalPath::FROM_CHILD &&
                distance(queryPos[tid], nodes[nearNode.nodeID].bounds) <
                    results[tid].dist) {
              curItem = nearNode;
              path = TraversalPath::FROM_PARENT;
              finishIter = true;
              // printf("Going to near node at: %d\n", curItem.nodeID);
            }
          } else {
            const uint32_t *leafList = leafLists+node.index;
            for (int i=0;i<node.count;i++) {
              const int   triID = leafList[i];
              const vec3i tri = faces[triID];
              vec3f closest;
              float nodeDist = pointTriDistance(queryPos[tid], vertices[tri[0]],
                                                vertices[tri[1]],
                                                vertices[tri[2]], closest);
              if (nodeDist < results[tid].dist) {
                results[tid].triID = triID;
                results[tid].point = closest;
                results[tid].dist = nodeDist;
              }
            }
          }

          if (!finishIter) {
            if (curItem.nodeID == 0) {
              break;
            } else {
              int parentID = node.parent;
              int siblingID = siblingNode(nodes, parentID, curItem.nodeID);
              float siblingDist =
                  distance(queryPos[tid], nodes[siblingID].bounds);
              bool atNearNode = curItem.distance < siblingDist ||
                                (curItem.nodeID > siblingID &&
                                 curItem.distance == siblingDist);

              if (path != TraversalPath::FROM_SIBLING && curItem.nodeID != 0 &&
                  atNearNode) {
                BVHCPATStackItem sibling;
                sibling.nodeID = siblingID;
                sibling.distance = siblingDist;
                if (sibling.distance < results[tid].dist) {
                  curItem = sibling;
                  path = TraversalPath::FROM_SIBLING;
                  // printf("Going to sibling at: %d\n", curItem.nodeID);
                  finishIter = true;
                }
              }

              if (!finishIter) {
                BVHCPATStackItem parent;
                parent.nodeID = parentID;
                parent.distance =
                    distance(queryPos[tid], nodes[parentID].bounds);
                curItem = parent;
                path = TraversalPath::FROM_CHILD;
                // printf("Going to parent at: %d\n", curItem.nodeID);
              }
            }
          }
        }
      }

      void BVH2EngineBT::doQueryGPU(const core::BVH2::Node *nodes,
                                    const uint32_t *leafLists,
                                    const vec3f *vertices, const vec3i *faces,
                                    const vec3f *queryPos, int numQueries,
                                    float maxQueryDistance,
                                    CPATResult *results) {
        int numThreads = 32;
        int numBlocks = (numQueries + numThreads - 1) / numThreads;
        BVH2BTQuery<<<numBlocks, numThreads>>>(nodes, leafLists, vertices,
                                               faces, queryPos, numQueries,
                                               maxQueryDistance, results);
        cudaDeviceSynchronize();
      }

      //void BVH2EnginePQ::doQuery(InMemoryCandidateHeap &candidates,
      //                         const BVH2Engine::PointData *model,
      //                         const vec3f &queryPos)
      //{
      //  std::priority_queue<std::pair<float,int>,
      //                      std::vector<std::pair<float,int>>,
      //                      std::greater<std::pair<float,int>>> pq;
      //  pq.push({0.f,0});
      //  while (!pq.empty()) {
      //    auto top = pq.top();
      //    pq.pop();

      //    const float currentR = candidates.currentRadius();
      //    if (top.first > currentR)
      //      break;

      //    const int nodeID = top.second;
      //    const BVH2::Node &node = model->bvh2.nodes[nodeID];
      //    
      //    if (node.count == 0) {
      //      const BVH2::Node &n0 = model->bvh2.nodes[node.index+0];
      //      const BVH2::Node &n1 = model->bvh2.nodes[node.index+1];
      //      const float d0 = distance(queryPos,n0.bounds);
      //      const float d1 = distance(queryPos,n1.bounds);
      //      if (d1 <= currentR)
      //        pq.push({d1,node.index+1});
      //      if (d0 <= currentR)
      //        pq.push({d0,node.index+0});
      //      continue;
      //    }

      //    const uint32_t *leafList = model->bvh2.leafLists.data()+node.index;
      //    for (int i=0;i<node.count;i++) {
      //      const int   pointID = leafList[i];
      //      const vec3f point   = model->points[pointID];
      //      float nodeDist = length(point - queryPos);
      //      maybe_insert(candidates, pointID, nodeDist);
      //    }
      //  }
      //  
      //  // finally, fill in unused k-nearest slots if we found
      //  // less than k entires
      //  for (int i = candidates.count; i < candidates.k; i++)
      //    candidates.memory[i] = {-1, candidates.maxRadius};
      //}


    } // ::gequel::knn::cpu
  } // ::gequel::knn
} // ::gequel
