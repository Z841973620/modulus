/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>


#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>

#include <limits>

#include "2dcommon.h"

const float Z_CENTER = (0.0f);

Shape contour_to_shape(std::vector<std::pair<float, float>> const &contour) {
  Shape result;
  if (contour.empty())
    return result;

  // add contour knots
  for (auto const &c : contour)
    result.knots.emplace_back(float3{c.first, c.second, 0.0f});

  float const half_width = 0.2f;
  auto add_vertices = [half_width, &result](float cx, float cy) {
    result.vertices.emplace_back(float3{cx, cy, Z_CENTER - half_width});
    result.vertices.emplace_back(float3{cx, cy, Z_CENTER + half_width});
  };

  unsigned int cur_vertex_id_base = 0;

  // add first vertices
  float cx = contour.front().first, cy = contour.front().second;
  add_vertices(cx, cy);

  for (size_t i = 1; i < contour.size(); ++i) {
    cx = contour[i].first;
    cy = contour[i].second;
    add_vertices(cx, cy);

    result.triangles.emplace_back(uint3{cur_vertex_id_base + 0,
                                        cur_vertex_id_base + 2,
                                        cur_vertex_id_base + 1});

    result.triangles.emplace_back(uint3{cur_vertex_id_base + 1,
                                        cur_vertex_id_base + 2,
                                        cur_vertex_id_base + 3});

    cur_vertex_id_base += 2;
  }

  return result;
}


void read_cli_file(std::string const &file_path, std::vector<Shape> &shapes) {
  std::ifstream file(file_path);
  if (!file) {
    throw std::runtime_error("Could not read file for reading '" + file_path +
                             "'");
  }

  auto ignore_line = [&file]() {
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  };

  // we need this since it seems those CLI files have a CRLF sequence at the
  // end of each line which leads to std::getline on linux to keep the CR in
  // the string
  // we're also skipping empty lines as they seem to appear sporadically in the
  // files provided
  auto safe_getline = [](std::ifstream &is, std::string &line) {
    do {
      std::getline(is, line);
      if (line.back() == '\r')
        line = line.substr(0, line.size() - 1);
    } while (line.empty());

    return !is.fail();
  };

  std::string cur_line;
  ignore_line(); // !pol
  ignore_line(); // v5 0.5

  int num_groups;
  safe_getline(file, cur_line);
  sscanf(cur_line.c_str(), "! %d", &num_groups);

  for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
    safe_getline(file, cur_line);
    float origin_x, origin_y;
    sscanf(cur_line.c_str(), "! %f %f", &origin_x, &origin_y);

    safe_getline(file, cur_line);
    int num_shapes;
    sscanf(cur_line.c_str(), "! %d", &num_shapes);

    safe_getline(file, cur_line);
    float bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y;
    sscanf(cur_line.c_str(), "! %f %f %f %f", &bbox_min_x, &bbox_min_y,
           &bbox_max_x, &bbox_max_y);

    std::vector<std::pair<float, float>> contour;

    for (int shape_idx = 0; shape_idx < num_shapes; ++shape_idx) {
      // read bounding box
      safe_getline(file, cur_line);
      float cur_bbox_min_x, cur_bbox_min_y, cur_bbox_max_x, cur_bbox_max_y;
      sscanf(cur_line.c_str(), "! %f %f %f %f", &cur_bbox_min_x,
             &cur_bbox_min_y, &cur_bbox_max_x, &cur_bbox_max_y);

      // read number of vertices
      safe_getline(file, cur_line);
      int cur_num_vertices;
      sscanf(cur_line.c_str(), "! %d", &cur_num_vertices);
      if (file.fail())
        throw std::runtime_error("failed at num vertices");

      // read vertices
      // while (contour.size() < static_cast<size_t>(cur_num_vertices + 1) &&
      //       file && safe_getline(file, cur_line) && cur_line != "NEXT") {
      //  float cur_pos_x, cur_pos_y;
      //  sscanf(cur_line.c_str(), "%f %f", &cur_pos_x, &cur_pos_y);
      //  contour.emplace_back(cur_pos_x, cur_pos_y);
      //}
      for (int i = 0; i < cur_num_vertices + 1; ++i) {
        if (!safe_getline(file, cur_line))
          throw std::runtime_error("error parsing file");

        float cur_pos_x, cur_pos_y;
        sscanf(cur_line.c_str(), "%f %f", &cur_pos_x, &cur_pos_y);
        contour.emplace_back(cur_pos_x, cur_pos_y);
      }

      safe_getline(file, cur_line);
      if (cur_line != "NEXT")
        throw std::runtime_error("expected NEXT statement");
      if (contour.size() != static_cast<size_t>(cur_num_vertices + 1))
        throw std::runtime_error("inconsistent number of vertices provided");

      shapes.push_back(contour_to_shape(contour));
      contour.clear();
    }
  }
}
