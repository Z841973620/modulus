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

namespace gequel {
  /*! core data structures, using across both host and device */
  namespace core {
        
    class BVH2 {
    public:
      struct Node {
        box3f bounds;
        uint32_t parent;
        uint32_t index;
        /*! if 0, childIdx is a inner child pointer; else it's a
          offset into item list, and count is the number of
          entires there */
        uint32_t count;
      };

      std::vector<Node>     nodes;
      /*! vector of all item lists; Node::childIdx points into this */
      std::vector<uint32_t> leafLists;
          
      void build(const std::vector<box3f> &primBounds);

    private:
      void buildRec(const std::vector<box3f> &primBounds,
                    int nodeID,
                    std::vector<uint32_t> &primIDs,
                    const box3f &treeBounds,
                    const box3f &centBounds);
    };
  }
}
