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

#include "gequel/common/common.h"

#include <vector>

namespace gequel {
  namespace core {

    class KDTree {
    public:
      struct Node {
        vec3f pos;
        uint32_t pointID : 30;
        uint32_t dim : 2;
      };

      static box3f computeBounds(const std::vector<vec3f>& points) {
        box3f bounds;
        for (auto pt : points)
          bounds.extend(pt);
        return bounds;
      }

      static std::vector<Node>
        buildTree(const std::vector<vec3f>& inputs, const box3f& bounds) {
        std::vector<Node> points(inputs.size());
        for (int pointID = 0; pointID < points.size(); pointID++) {
          points[pointID].pos = inputs[pointID];
          points[pointID].pointID = pointID;
        }
        std::vector<Node> tree(inputs.size());
        kdTreeBuildRec(tree, points, bounds, 0, 0, 0, points.size());

        // checkTree(tree,bounds,0);
        return tree;
      }

    private:
      template<int DimIdx>
      struct VecComparator {
        inline bool operator()(const Node& a, const Node& b) const;
      };
      static inline void kdTreeBuildRec(std::vector<Node>& tree,
        std::vector<Node>& points,
        const box3f& bounds, size_t node,
        int level, size_t begin, size_t end);

      static inline void checkTree(const std::vector<Node>& nodes,
        const box3f& bounds, int nodeID);
    };


    template<>
    struct KDTree::VecComparator<0> {
      inline bool operator()(const Node& a, const Node& b) const {
        return a.pos.x < b.pos.x;
      }
    };

    template<>
    struct KDTree::VecComparator<1> {
      inline bool operator()(const Node& a, const Node& b) const {
        return a.pos.y < b.pos.y;
      }
    };

    template<>
    struct KDTree::VecComparator<2> {
      inline bool operator()(const Node& a, const Node& b) const {
        return a.pos.z < b.pos.z;
      }
    };


    void KDTree::checkTree(const std::vector<Node>& nodes,
      const box3f& bounds, int nodeID) {
      if (nodeID >= nodes.size())
        return;

      if (!bounds.contains(nodes[nodeID].pos)) {
        PING;
        PRINT(bounds);
        PRINT(nodeID);
        PRINT(nodes[nodeID].pos);
        throw std::runtime_error("not a valid kd-tree!!!!");
      }

      box3f lBounds = bounds;
      box3f rBounds = bounds;
      const Node& node = nodes[nodeID];
      lBounds.upper[node.dim] = rBounds.lower[node.dim] =
        node.pos[node.dim];
      checkTree(nodes, lBounds, 2 * nodeID + 1);
      checkTree(nodes, rBounds, 2 * nodeID + 2);
    }

    inline void KDTree::kdTreeBuildRec(std::vector<Node>& tree,
      std::vector<Node>& points,
      const box3f& bounds, size_t node,
      int level, size_t begin, size_t end) {
      if (node >= points.size())
        return;

      size_t numInThisSubtree = end - begin;
      if (numInThisSubtree == 1) {
        tree[node] = points[begin];
        return;
      }
      if (numInThisSubtree < 1)
        return;

      // std::cout << "splitting node " << node << " range " << begin << "
      // .. " << end << " "
      //           << bounds << std::endl;
      // TODO: make boxes by trimming, not by computing tight bboxes
      int splitDim = arg_max(bounds.size());
      // PRINT(splitDim);
      switch (splitDim) {
      case 0:
        std::sort(points.begin() + begin, points.begin() + end,
          VecComparator<0>());
        break;
      case 1:
        std::sort(points.begin() + begin, points.begin() + end,
          VecComparator<1>());
        break;
      case 2:
      default:
        std::sort(points.begin() + begin, points.begin() + end,
          VecComparator<2>());
        break;
      }
      size_t depthOfSubtree
        //_BitScanReverse64 on windows? __clzz in cuda?
        = 64 - __builtin_clzll(numInThisSubtree);
      // PRINT(numInThisSubtree);
      // PRINT(depthOfSubtree);
      size_t numChildLevels = depthOfSubtree - 1;

      size_t leftFull = begin + (1 << numChildLevels) - 1;
      size_t rightEmpty =
        end - (numChildLevels == 0 ? 0 : (1 << (numChildLevels - 1)));

      size_t pMedianPos = std::min(size_t(leftFull), size_t(rightEmpty));
      // PRINT(pMedianPos);
      tree[node] = points[pMedianPos];
      tree[node].dim = splitDim;
      box3f lBounds = bounds;
      box3f rBounds = bounds;
      lBounds.upper[splitDim] = rBounds.lower[splitDim] =
        tree[node].pos[splitDim];
#if 1
      owl::parallel_for(2, [&](int side) {
        if (side)
          kdTreeBuildRec(tree, points, lBounds, 2 * node + 1, level + 1,
            begin, pMedianPos);
        else
          kdTreeBuildRec(tree, points, rBounds, 2 * node + 2, level + 1,
            pMedianPos + 1, end);
        });
#else
      kdTreeBuildRec(tree, points, lBounds, 2 * node + 1, level + 1,
        begin, pMedianPos);
      kdTreeBuildRec(tree, points, rBounds, 2 * node + 2, level + 1,
        pMedianPos + 1, end);
#endif
    }

  }
}
