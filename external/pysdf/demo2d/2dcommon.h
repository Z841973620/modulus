#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>

struct Shape {
  std::vector<float3> knots;
  std::vector<float3> vertices;
  std::vector<uint3> triangles;
};

void read_cli_file(std::string const &file_path, std::vector<Shape> &shapes);

