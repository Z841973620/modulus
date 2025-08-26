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

#include "owl/common/math/AffineSpace.h"

#include <vector>
#include <iostream>
#include <fstream>

namespace gequel {
  // p4 reader
  inline std::vector<vec3f> readP4(const char* filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
      std::cout << "file " << filename << " does not exist\n";
      return {};
    }
    std::vector<vec3f> points;
    char buffer[16];
    while (true) {
      f.read(buffer, 16);
      if (f.eof()) {
        break;
      }

      vec3f p;
      p.x = *((float*)buffer);
      p.y = *((float*)(buffer + 4));
      p.z = *((float*)(buffer + 8));
      points.push_back(p);
    }
    return points;
  }

  // p4 writer
  inline void writeP4(const char* filename, const std::vector<vec3f>& points) {
    std::ofstream f(filename, std::ios::binary);
    char buffer[16];
    for (const vec3f& p : points) {
      *((float*)buffer) = p.x;
      *((float*)(buffer + 4)) = p.y;
      *((float*)(buffer + 8)) = p.z;
      *((float*)(buffer + 12)) = 0;
      f.write(buffer, 16);
    }
  }

} // namespace gequel
