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

#include "gequel/common/cpu/datastructures/bvh2.h"

namespace gequel {
  namespace core {
        
    void BVH2::buildRec(const std::vector<box3f> &primBounds,
                       int nodeID,
                       std::vector<uint32_t> &primIDs,
                       const box3f &treeBounds,
                       const box3f &centBounds)
    {
      Node &node = nodes[nodeID];
      node.bounds = treeBounds;
      if (length(centBounds.size()) < 1e-3) {
        node.count  = primIDs.size();
        node.index  = leafLists.size();
        for (auto prim : primIDs)
          leafLists.push_back(prim);
        primIDs.clear();
        return;
      }

      std::vector<uint32_t> lPrims, rPrims;
      box3f lTreeBounds;
      box3f rTreeBounds;
      box3f lCentBounds;
      box3f rCentBounds;
      int dim = arg_max(centBounds.size());
      float where = centBounds.center()[dim];
      for (auto primID : primIDs) {
        const box3f pb = primBounds[primID];
        const vec3f pc = pb.center();
        if (pc[dim] < where) {
          lTreeBounds.extend(pb);
          lCentBounds.extend(pc);
          lPrims.push_back(primID);
        } else {
          rTreeBounds.extend(pb);
          rCentBounds.extend(pc);
          rPrims.push_back(primID);
        }
      }
      assert(!lPrims.empty());
      assert(!rPrims.empty());
      primIDs.clear();
      // NO parallel_for here, the code above isn't thread safe:

      int childID = nodes.size();
      node.index = childID;
      node.count = 0;
      // careful: 'Node &node' invalid from here on:
      nodes.emplace_back();
      nodes.back().parent = nodeID;
      nodes.emplace_back();
      nodes.back().parent = nodeID;
      buildRec(primBounds,childID+0,lPrims,lTreeBounds,lCentBounds);
      buildRec(primBounds,childID+1,rPrims,rTreeBounds,rCentBounds);
    }
    
    void BVH2::build(const std::vector<box3f> &primBounds)
    {
      assert(!primBounds.empty());
      box3f worldCentBounds;
      box3f worldPrimBounds;
      std::vector<uint32_t> primIDs;
      for (auto pb : primBounds) {
        worldPrimBounds.extend(pb);
        worldCentBounds.extend(pb.center());
        primIDs.push_back(primIDs.size());
      }

      nodes.emplace_back();
      nodes.back().parent = 0; // arbitrary value - shouldn't follow parent at root
      buildRec(primBounds,0,primIDs,worldPrimBounds,worldCentBounds);
    }
      
  }
}
