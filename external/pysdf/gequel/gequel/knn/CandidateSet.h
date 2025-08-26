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

#include "../knn/knn.h"

namespace gequel {
  namespace knn {

    /*! strcut that maintains the k nearest candidates during
      traversal.  note this class is "kind of" abstract, with
      differnet implemnettations below, but will _not_ be made virtual
      - we'll rely on whoever uses it to use it as a template paramters */
    struct CandidateSet
    {
      inline GQL_BOTH CandidateSet(float maxRadius);
      inline GQL_BOTH float currentRadius() const;
      inline GQL_BOTH int   currentCount() const;
      inline GQL_BOTH int   maxCount() const;
      
      /*! insert a new candidate into the candidate set. this shold
          ONLY ever get called for candidates that are GUARANTEED to
          be <= currentRadius() */
      inline GQL_BOTH void  insert(int nodeID, float distance);
    };

    template<typename CandidateSet>
    /*! test if candidate should be inserted, and if so, do it; else
      ignore. returns true if inserted, false otherwise */
    inline GQL_BOTH bool maybe_insert(CandidateSet &candidates, int nodeID, float distance)
    {
      if (distance >= candidates.currentRadius()) return false;
      candidates.insert(nodeID, distance);
      return true;
    }
    
    
    /*! Implements a candidate set that lives in memory
        allocated/maintained by somebody else */
    struct InMemoryCandidateHeap /* implmenets CandidateSet */
    {
      inline GQL_BOTH InMemoryCandidateHeap(KNNResult *const memory,
                                            int k,
                                            float maxRadius=1e20f);
      inline GQL_BOTH float currentRadius() const;
      inline GQL_BOTH int   currentCount() const;
      inline GQL_BOTH int   maxCount() const;
      inline GQL_BOTH void  insert(int nodeID, float distance);
      /*! test if candidate should be inserted, and if so, do it; else
          ignore. returns true if inserted, false otherwise */
      inline GQL_BOTH bool  maybe_insert(int nodeID, float distance);

      KNNResult *const memory;
      const int   k;
      const float maxRadius;
      int   count { 0 };
    };



    /*! Implements a candidate set with given max size that can then
        be put into registers */
    template<int MAX_K>
    struct MaxSizeCandidateHeap /* implements CandidateSet */
    {
      inline GQL_BOTH MaxSizeCandidateHeap(KNNResult *const memory,
                                             int k,
                                             float maxRadius=1e20f);
      inline GQL_BOTH ~MaxSizeCandidateHeap();
      inline GQL_BOTH float currentRadius() const;
      inline GQL_BOTH int   currentCount() const;
      inline GQL_BOTH int   maxCount() const;
      inline GQL_BOTH void  insert(int nodeID, float distance);
      /*! test if candidate should be inserted, and if so, do it; else
          ignore. returns true if inserted, false otherwise */
      inline GQL_BOTH bool  maybe_insert(int nodeID, float distance);

      KNNResult candidates[MAX_K];
      KNNResult *const writeBackMemory;
      const int   k;
      const float maxRadius;
      int   count { 0 };
    };

    template<>
    struct MaxSizeCandidateHeap<2> /* implements CandidateSet */
    {
      inline GQL_BOTH MaxSizeCandidateHeap(KNNResult *const memory,
                                             int k,
                                             float maxRadius=1e20f);
      inline GQL_BOTH ~MaxSizeCandidateHeap();
      inline GQL_BOTH float currentRadius() const;
      inline GQL_BOTH int   currentCount() const;
      inline GQL_BOTH int   maxCount() const;
      inline GQL_BOTH void  insert(int nodeID, float distance);
      /*! test if candidate should be inserted, and if so, do it; else
          ignore. returns true if inserted, false otherwise */
      inline GQL_BOTH bool  maybe_insert(int nodeID, float distance);

      KNNResult candidate0, candidate1;
      KNNResult *const writeBackMemory;
      const float maxRadius;
      const int count{ 2 };
    };

    template<>
    struct MaxSizeCandidateHeap<4> /* implements CandidateSet */
    {
      inline GQL_BOTH MaxSizeCandidateHeap(KNNResult *const memory,
                                             int k,
                                             float maxRadius=1e20f);
      inline GQL_BOTH ~MaxSizeCandidateHeap();
      inline GQL_BOTH float currentRadius() const;
      inline GQL_BOTH int   currentCount() const;
      inline GQL_BOTH int   maxCount() const;
      inline GQL_BOTH void  insert(int nodeID, float distance);
      /*! test if candidate should be inserted, and if so, do it; else
          ignore. returns true if inserted, false otherwise */
      inline GQL_BOTH bool  maybe_insert(int nodeID, float distance);

      KNNResult candidate0, candidate1, candidate2, candidate3;
      KNNResult *const writeBackMemory;
      const float maxRadius;
      const int count{ 4 };
    };



    inline GQL_BOTH
    InMemoryCandidateHeap::InMemoryCandidateHeap(KNNResult *const memory,
                                                 int k,
                                                 float maxRadius)
      : memory(memory),
        k(k),
        maxRadius(maxRadius),
        count(0)
    {}
    
    inline GQL_BOTH
    float InMemoryCandidateHeap::currentRadius() const
    {
      return count < k ? maxRadius : memory[0].dist;
    }
    
    inline GQL_BOTH
    int   InMemoryCandidateHeap::currentCount() const
    {
      return count;
    }
    
    inline GQL_BOTH
    int   InMemoryCandidateHeap::maxCount() const
    {
      return k;
    }

    inline GQL_BOTH
    void  InMemoryCandidateHeap::insert(int nodeID, float dist)
    {
      assert(dist <= currentRadius());

      int pos = 0;
      if (count < k) {
        pos = count++;
        while (pos > 0) {
          int parent = (pos-1)>>1;
          if (memory[parent].dist >= dist)
            break;
          
          memory[pos] = memory[parent];
          pos = parent;
          continue;
        }
      } else {
        pos = 0;
        while (1) {
          const int lChild = 2*pos+1;
          if (lChild >= count)
            break;
          int biggestChild = lChild;
          
          const int rChild = lChild+1;
          if (rChild < count && memory[rChild].dist > memory[lChild].dist)
            biggestChild = rChild;
          if (memory[biggestChild].dist <= dist)
            break;
          memory[pos] = memory[biggestChild];
          pos = biggestChild;
          continue;
        }
      }
      memory[pos] = KNNResult(nodeID,dist);
    }






    template<int MAX_K>
    inline GQL_BOTH
    MaxSizeCandidateHeap<MAX_K>::MaxSizeCandidateHeap(KNNResult *const memory,
                                                          int k,
                                                          float maxRadius)
      : writeBackMemory(memory),
        k(k),
        maxRadius(maxRadius),
        count(0)
    {
    }

    inline GQL_BOTH
    MaxSizeCandidateHeap<2>::MaxSizeCandidateHeap(KNNResult *const memory,
                                                          int k,
                                                          float maxRadius)
      : writeBackMemory(memory),
        maxRadius(maxRadius)
    {
      candidate0 = candidate1 = KNNResult(-1, maxRadius);
    }

    inline GQL_BOTH
    MaxSizeCandidateHeap<4>::MaxSizeCandidateHeap(KNNResult *const memory,
                                                          int k,
                                                          float maxRadius)
      : writeBackMemory(memory),
        maxRadius(maxRadius)
    {
      candidate0 = candidate1 = candidate2 = candidate3 = KNNResult(-1, maxRadius);
    }

    template<int MAX_K>
    inline GQL_BOTH
    MaxSizeCandidateHeap<MAX_K>::~MaxSizeCandidateHeap()
    {
      for (int i=0;i<count;i++)
        writeBackMemory[i] = candidates[i];
    }

    inline GQL_BOTH
    MaxSizeCandidateHeap<2>::~MaxSizeCandidateHeap()
    {
      writeBackMemory[0] = candidate0;
      writeBackMemory[1] = candidate1;
    }

    inline GQL_BOTH
    MaxSizeCandidateHeap<4>::~MaxSizeCandidateHeap()
    {
      writeBackMemory[0] = candidate0;
      writeBackMemory[1] = candidate1;
      writeBackMemory[2] = candidate2;
      writeBackMemory[3] = candidate3;
    }
    
    template<int MAX_K>
    inline GQL_BOTH
    float MaxSizeCandidateHeap<MAX_K>::currentRadius() const
    {
      return count < k ? maxRadius : candidates[0].dist;
    }
    
    inline GQL_BOTH
    float MaxSizeCandidateHeap<2>::currentRadius() const
    {
      return candidate1.dist;
    }
    
    inline GQL_BOTH
    float MaxSizeCandidateHeap<4>::currentRadius() const
    {
      return candidate3.dist;
    }
    
    template<int MAX_K>
    inline GQL_BOTH
    int   MaxSizeCandidateHeap<MAX_K>::currentCount() const
    {
      return count;
    }
    
    inline GQL_BOTH
    int   MaxSizeCandidateHeap<2>::currentCount() const
    {
      return 2;
    }
    
    inline GQL_BOTH
    int   MaxSizeCandidateHeap<4>::currentCount() const
    {
      return 4;
    }
    
    template<int MAX_K>
    inline GQL_BOTH
    int   MaxSizeCandidateHeap<MAX_K>::maxCount() const
    {
      return k;
    }
    
    inline GQL_BOTH
    int   MaxSizeCandidateHeap<2>::maxCount() const
    {
      return 2;
    }
    
    inline GQL_BOTH
    int   MaxSizeCandidateHeap<4>::maxCount() const
    {
      return 4;
    }

    template<int MAX_K>
    inline GQL_BOTH
    void  MaxSizeCandidateHeap<MAX_K>::insert(int nodeID, float dist)
    {
      assert(dist <= currentRadius());

      int pos = 0;
      if (count < k) {
        pos = count++;
        while (pos > 0) {
          int parent = (pos-1)>>1;
          if (candidates[parent].dist >= dist)
            break;
          
          candidates[pos] = candidates[parent];
          pos = parent;
          continue;
        }
      } else {
        pos = 0;
        while (1) {
          const int lChild = 2*pos+1;
          if (lChild >= count)
            break;
          int biggestChild = lChild;
          
          const int rChild = lChild+1;
          if (rChild < count && candidates[rChild].dist > candidates[lChild].dist)
            biggestChild = rChild;
          if (candidates[biggestChild].dist <= dist)
            break;
          candidates[pos] = candidates[biggestChild];
          pos = biggestChild;
          continue;
        }
      }
      candidates[pos] = KNNResult(nodeID,dist);
    }

    inline GQL_BOTH
    void  MaxSizeCandidateHeap<2>::insert(int nodeID, float dist)
    {
      assert(dist <= currentRadius());

      if (dist < candidate0.dist) {
        candidate1 = candidate0;
        candidate0 = KNNResult(nodeID, dist);
      } else if (dist < candidate1.dist) {
        candidate1 = KNNResult(nodeID, dist);
      }
    }

    inline GQL_BOTH
    void  MaxSizeCandidateHeap<4>::insert(int nodeID, float dist)
    {
      assert(dist <= currentRadius());

      if (dist < candidate0.dist) {
        candidate3 = candidate2;
        candidate2 = candidate1;
        candidate1 = candidate0;
        candidate0 = KNNResult(nodeID, dist);
      } else if (dist < candidate1.dist) {
        candidate3 = candidate2;
        candidate2 = candidate1;
        candidate1 = KNNResult(nodeID, dist);
      } else if (dist < candidate2.dist) {
        candidate3 = candidate2;
        candidate2 = KNNResult(nodeID, dist);
      } else if (dist < candidate3.dist) {
        candidate3 = KNNResult(nodeID, dist);
      }
    }
  }
}
