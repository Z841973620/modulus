
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

#include "gequel/knn/gpu/KDTree.h"
#include "gequel/common/gpu/datastructures/stack.h"

#include <device_launch_parameters.h>

namespace gequel {
  /* find closest points */
  namespace knn {
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

      struct KNNStackItem {
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
                                 int numQueries, int k, float maxQueryDistance,
                                 KNNResult *results) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= numQueries)
          return;

        InMemoryCandidateHeap candidates(results + k * tid, k, maxQueryDistance);
        TraversalPath path = TraversalPath::FROM_PARENT;
        KNNStackItem curItem;
        curItem.nodeID = 0;
        curItem.bounds = bounds;

        // TODO: short stack may be necessary for better performance
        while (curItem.nodeID >= 0) {
          if (path == TraversalPath::FROM_CHILD && curItem.nodeID == 0) {
            break;
          }
          //printf("Currently at: %d; traversal path: %d\n", curItem.nodeID, path);
          const KDTree::Tree::Node &node = nodes[curItem.nodeID];

          // TODO: Add fast path here in case node is too far and was taken off a
          // stack

          //PRINT(node.pos);
          float nodeDist = length(node.pos - queryPos[tid]);
          //PRINT(nodeDist);
          //PRINT(candidates.currentRadius());
          //PRINT(candidates.currentCount());
          if (path != TraversalPath::FROM_CHILD) {
            maybe_insert(candidates, node.pointID, nodeDist);
          }

          KNNStackItem leftItem, rightItem;
          rightItem.bounds = leftItem.bounds = curItem.bounds;
          leftItem.bounds.upper[node.dim] = rightItem.bounds.lower[node.dim] = node.pos[node.dim];
          leftItem.nodeID = 2 * curItem.nodeID + 1;
          rightItem.nodeID = 2 * curItem.nodeID + 2;
          bool onLeft = queryPos[tid][node.dim] < node.pos[node.dim];
          const KNNStackItem &nearNode = onLeft ? leftItem : rightItem;

          bool finishIter = false;
          int siblingID = siblingNode(curItem.nodeID);
          int parentID = parentNode(curItem.nodeID);
          bool atNearNode = true;
          if (curItem.nodeID != 0) {
            int parentSplitDim = nodes[parentID].dim;
            atNearNode = curItem.nodeID % 2 == 1 ^
                         queryPos[tid][parentSplitDim] >=
                             nodes[parentID].pos[parentSplitDim];
          }
          if (path != TraversalPath::FROM_CHILD &&
              nearNode.nodeID < numPoints &&
              distance(queryPos[tid], nearNode.bounds) <
                  candidates.currentRadius()) {
            curItem = nearNode;
            path = TraversalPath::FROM_PARENT;
            finishIter = true;
            //printf("Going to near node at: %d\n", curItem.nodeID);
          } else if (path != TraversalPath::FROM_SIBLING &&
                     curItem.nodeID != 0 && siblingID < numPoints &&
                     atNearNode) {
            KNNStackItem sibling;
            sibling.nodeID = siblingID;
            sibling.bounds =
                computeBounds(nodes, numPoints, bounds, sibling.nodeID);
            if (distance(queryPos[tid], sibling.bounds) < candidates.currentRadius()) {
              curItem = sibling;
              path = TraversalPath::FROM_SIBLING;
              //printf("Going to sibling at: %d\n", curItem.nodeID);
              finishIter = true;
            }
          }
          if (!finishIter) {
            KNNStackItem parent;
            parent.nodeID = parentID;
            parent.bounds =
                computeBounds(nodes, numPoints, bounds, parent.nodeID);
            curItem = parent;
            path = TraversalPath::FROM_CHILD;
            //printf("Going to parent at: %d\n", curItem.nodeID);
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
        CUDA_PTR_DEVICE(resultArray);
        CUDA_PTR_DEVICE(queries);

        KDTree::PointData *model = (KDTree::PointData *)_model;
        int numThreads = 1024;
        int numBlocks = (numQueries + numThreads - 1) / numThreads;

        int numPoints = model->nodes.size();

        doQueryGPU<<<numBlocks, numThreads>>>(
            thrust::raw_pointer_cast(&model->nodes[0]), numPoints,
            model->bounds, queries, numQueries, k, maxQueryDistance,
            resultArray);

        cudaDeviceSynchronize();
      }

    } // ::gequel::knn::gpu
  } // ::gequel::knn
} // ::gequel
