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

#include "gequel/fcp/gpu/KDTree.h"
#include "gequel/common/gpu/datastructures/stack.h"

#include <device_launch_parameters.h>

namespace gequel {
  /* find closest points */
  namespace fcp {
    namespace gpu {
      thrust::device_vector<KDTree::Tree::Node> KDTree::PointData::buildTree(const std::vector<vec3f>& points, const box3f& bounds) {
        std::vector<Tree::Node> tree = Tree::buildTree(points, bounds);
        thrust::device_vector<Tree::Node> d_tree(tree.begin(), tree.end());
        return d_tree;
      }

      abstract::PointData *KDTree::createModel(const std::vector<vec3f> &points)
      {
        return new KDTree::PointData(points);
      }

      struct FCPStackItem {
        int nodeID;
        box3f bounds;
      };

      inline __device__ int nodeLevel(size_t n) {
        return 64 - __clzll(n + 1) - 1;
      }

      // Returns whether or not x is a child node of y
      inline __device__ bool isChild(size_t x, size_t y) {
        if (x < y) {
          return false;
        }
        size_t levelx = nodeLevel(x);
        size_t levely = nodeLevel(y);
        int k = levelx - levely;
        size_t minChild = (y << k) + (1 << k) - 1;
        size_t maxChild = (y << k) + (1 << k+1) - 2;
        return minChild <= x && x <= maxChild;
      }

      inline __device__ size_t siblingNode(size_t n) {
        return (int)n + 2 * ((int)n & 1) - 1; // +1 if odd (left child), -1 if even (right child)
      }

      inline __device__ size_t parentNode(size_t n) {
        return (n - 1) >> 1;
      }

      // "Slow path" for computing bbox bounds. Need to do this when going to a
      // parent/sibling not on the stack.
      inline __device__ box3f computeBounds(const KDTree::Tree::Node *nodes,
                                            int numPoints, box3f bounds,
                                            int x) {
        int idx = 0;
        while (idx < x) {
          int leftChild = 2 * idx + 1;
          int rightChild = leftChild + 1;
          int splitDim = nodes[idx].dim;
          if (isChild(x, leftChild)) {
            bounds.upper[splitDim] = nodes[idx].pos[splitDim];
            idx = leftChild;
          } else {
            bounds.lower[splitDim] = nodes[idx].pos[splitDim];
            idx = rightChild;
          }
        }
        return bounds;
      }

      enum TraversalPath {
        FROM_PARENT,
        FROM_SIBLING,
        FROM_CHILD,
      };

      __global__ void doQueryGPU(const KDTree::Tree::Node *nodes, int numPoints,
                                 const box3f bounds, const vec3f *queryPos,
                                 int numQueries, float maxQueryDistance,
                                 FCPResult *results) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= numQueries)
          return;

        float dist = maxQueryDistance;
        int nodeID = -1;

        // TODO: short stack may be necessary for better performance

        TraversalPath path = TraversalPath::FROM_PARENT;
        FCPStackItem curItem;
        curItem.nodeID = 0;
        curItem.bounds = bounds;
        while (curItem.nodeID >= 0) {
          if (path == TraversalPath::FROM_CHILD && curItem.nodeID == 0) {
            break;
          }
          //printf("Currently at: %d; traversal path: %d\n", curItem.nodeID, path);
          const KDTree::Tree::Node &node = nodes[curItem.nodeID];

          // TODO: Add fast path here in case node is too far and was taken off a
          // stack

          float nodeDist = length(node.pos - queryPos[i]);
          if (path != TraversalPath::FROM_CHILD && nodeDist < dist) {
            dist = nodeDist;
            nodeID = node.pointID;
            //printf("current distance: %f@%d\n", dist, nodeID);
          }

          FCPStackItem leftItem, rightItem;
          rightItem.bounds = leftItem.bounds = curItem.bounds;
          leftItem.bounds.upper[node.dim] = rightItem.bounds.lower[node.dim] =
              node.pos[node.dim];
          leftItem.nodeID = 2 * curItem.nodeID + 1;
          rightItem.nodeID = 2 * curItem.nodeID + 2;
          bool onLeft = queryPos[i][node.dim] < node.pos[node.dim];
          const FCPStackItem &nearNode = onLeft ? leftItem : rightItem;

          bool finishIter = false;
          int siblingID = siblingNode(curItem.nodeID);
          int parentID = parentNode(curItem.nodeID);
          bool atNearNode = true;
          if (curItem.nodeID != 0) {
            int parentSplitDim = nodes[parentID].dim;
            atNearNode = curItem.nodeID % 2 == 1 ^
                         queryPos[i][parentSplitDim] >=
                             nodes[parentID].pos[parentSplitDim];
          }
          if (path != TraversalPath::FROM_CHILD &&
              nearNode.nodeID < numPoints &&
              distance(queryPos[i], nearNode.bounds) < dist) {
            curItem = nearNode;
            path = TraversalPath::FROM_PARENT;
            finishIter = true;
            //printf("Going to near node at: %d\n", curItem.nodeID);
          } else if (path != TraversalPath::FROM_SIBLING &&
                     curItem.nodeID != 0 && siblingID < numPoints &&
                     atNearNode) {
            FCPStackItem sibling;
            sibling.nodeID = siblingID;
            sibling.bounds =
                computeBounds(nodes, numPoints, bounds, sibling.nodeID);
            if (distance(queryPos[i], sibling.bounds) < dist) {
              curItem = sibling;
              path = TraversalPath::FROM_SIBLING;
              //printf("Going to sibling at: %d\n", curItem.nodeID);
              finishIter = true;
            }
          }
          if (!finishIter) {
            FCPStackItem parent;
            parent.nodeID = parentID;
            parent.bounds =
                computeBounds(nodes, numPoints, bounds, parent.nodeID);
            curItem = parent;
            path = TraversalPath::FROM_CHILD;
            //printf("Going to parent at: %d\n", curItem.nodeID);
          }
        }

        results[i].distance = dist;
        results[i].pointID = nodeID;
      }

      void KDTree::findClosestPoint(FCPResult *results, int numQueries,
                                    abstract::PointData *_model,
                                    const vec3f *queries,
                                    const float maxQueryDistance)
      {
        CUDA_PTR_DEVICE(results);
        CUDA_PTR_DEVICE(queries);
        int numThreads = 1024;
        int numBlocks = (numQueries + numThreads - 1) / numThreads;

        assert(_model);
        KDTree::PointData *model = (KDTree::PointData *)_model;
        int numPoints = model->nodes.size();

        doQueryGPU<<<numBlocks, numThreads>>>(
            thrust::raw_pointer_cast(&model->nodes[0]), numPoints,
            model->bounds, queries, numQueries,
            maxQueryDistance, results);

        cudaDeviceSynchronize();
      }
    }
  }
}
