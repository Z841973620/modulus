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

namespace gequel {
  namespace common {
    namespace gpu {
      namespace datastructures {

        inline __device__ void stackInit(int* head) {
          *head = -1;
        }

        template<typename T>
        inline __device__ void stackPush(T* stack, int* head, const T& item) {
          (*head)++;
          stack[*head] = item;
        }

        template<typename T>
        inline __device__ T stackPop(T* stack, int* head) {
          T item = stack[*head];
          (*head)--;
          return item;
        }

      }
    }
  }
}
