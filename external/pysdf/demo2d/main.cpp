#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <sdf_cpp.h>

#include <cuda_runtime.h>

#include <timing.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
using std::vector;

#include "2dcommon.h"

#define CUDA_CHECK(A)                                     \
  do {                                                    \
    cudaError_t err = A;                                  \
    if (err != cudaSuccess) {                             \
      throw std::runtime_error(cudaGetErrorString(err));  \
    }                                                     \
  } while (false)


//! Create regular 3D grid in VTK with "grid_values" as cell values.
void write_grid2d_vtk(const std::string& filename, float3 origin, float dx, float dy, float z, int Nx, int Ny,
                      const std::vector<float>& grid_values) {
  nvtxRangePush("write_grid2d_vtk");
  std::ofstream outFile(filename);

  outFile << "# vtk DataFile Version 2.0\n";
  outFile << "Grid\n";
  outFile << "ASCII\n";
  outFile << "DATASET STRUCTURED_GRID\n";
  outFile << "DIMENSIONS " << Nx << " " << Ny  << " 1"<< std::endl;
  outFile << "POINTS " << (Nx)*(Ny) << " float\n";
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      outFile << origin.x + i * dx << " " << origin.y + j * dy << " "
              << z << "\n";
    }
  }

  outFile << "POINT_DATA " << grid_values.size() << "\n";
  outFile << "SCALARS scalars double 1\n";
  outFile << "LOOKUP_TABLE default\n";
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      outFile << grid_values[Ny*i + j] << "\n";
    }
  }
  nvtxRangePop();
}

void write_ribbon_vtk(std::string filename, const std::vector<float3>& vertices, const std::vector<uint3>& triangles) {
  nvtxRangePush("write_ribbon_vtk");
  std::ofstream outFile(filename);
  outFile << "# vtk DataFile Version 2.0\n";
  outFile << "Ribbon\n";
  outFile << "ASCII\n";
  outFile << "DATASET UNSTRUCTURED_GRID\n";
  outFile << "POINTS " << vertices.size() << " float\n";
  for(auto& v : vertices) {
    outFile << v.x << " " << v.y << " " << v.z << "\n";
  }
  outFile << "CELLS " << triangles.size() << " " << triangles.size() * 4 << "\n";
  for(auto& t : triangles) {
    outFile << "3 " << t.x << " " << t.y << " " << t.z << "\n";
  }
  outFile << "CELL_TYPES " << triangles.size() << "\n";
  for(auto& t : triangles) {
    outFile << "5\n";
  }
  nvtxRangePop();
}


// Reads mesh 'filename' and returns vertices that are grouped per triangle.
bool loadMesh(const std::string& filename, vector<double>& vertices, float3& mesh_min, float3& mesh_max) {

  std::ifstream mesh_in(filename);
  if(!mesh_in) {
    throw std::runtime_error("ERROR: Couldn't open mesh file " + filename);
  }

  vector<tinyobj::shape_t> shapes;
  vector<tinyobj::material_t> materials;
  tinyobj::attrib_t attrib;
  std::string warn;
  std::string err;

  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              filename.c_str());

  if(!ret) {
    throw std::runtime_error("ERROR/WARNING loading OBJ. \nWARNING:" + warn + "\n ERROR:" + err);
  }

  float fmax = std::numeric_limits<float>::max();
  float flowest = std::numeric_limits<float>::lowest();

  mesh_max = make_float3(flowest, flowest, flowest);
  mesh_min = make_float3(fmax, fmax, fmax);

  for (size_t s = 0; s < shapes.size(); s++) {
    size_t index_offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      int fv = shapes[s].mesh.num_face_vertices[f];

      if (fv == 3) {
        uint32_t vertexOffset = (uint32_t)vertices.size();

        for (size_t v = 0; v < fv; v++) {
          // access to vertex
          tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

          if (idx.vertex_index >= 0) {
            double vx = attrib.vertices[3 * idx.vertex_index + 0];
            double vy = attrib.vertices[3 * idx.vertex_index + 1];
            double vz = attrib.vertices[3 * idx.vertex_index + 2];
            
            vertices.push_back(vx);
            vertices.push_back(vy);
            vertices.push_back(vz);

            // update mesh bounds
            mesh_min.x = std::min((float)vx, mesh_min.x);
            mesh_min.y = std::min((float)vy, mesh_min.y);
            mesh_min.z = std::min((float)vz, mesh_min.z);

            mesh_max.x = std::max((float)vx, mesh_max.x);
            mesh_max.y = std::max((float)vy, mesh_max.y);
            mesh_max.z = std::max((float)vz, mesh_max.z);

          }
        }
        index_offset += fv;

      }
    }
  }

  return true;
}


int main(int argc, char** argv) {
  std::cout << "HELLO\n";
  std::string filename;

  if(argc == 1) {
    printf("Usage: sdf [-h] [scene.cli]");
    return EXIT_FAILURE;
  }
  if(argc > 1) {
    if(std::string(argv[1]) == std::string("-h")) {
      printf("Usage: sdf [-h] [scene.cli]");
      return EXIT_FAILURE;
    }
    else {
      filename = std::string(argv[1]);
    }
  }
  
  // load polygons
  std::vector<Shape> shapes;
  nvtxRangePush("Load CLI");
  read_cli_file(filename, shapes);
  nvtxRangePop();

  
  // compact all shapes into one single primitive
  std::vector<float3> vertices3d;
  std::vector<uint3> triangles3d;
  auto max_num = std::numeric_limits<float>::max();
  auto min_num = std::numeric_limits<float>::lowest();
  float3 mesh_min = make_float3(max_num, max_num, max_num);
  float3 mesh_max = make_float3(min_num, min_num, min_num);
  {
    unsigned int offset = 0;
    for (auto const &s : shapes) {
      for (auto const &v : s.vertices) {
        vertices3d.push_back(v);
        // update mesh bounds
        mesh_min.x = std::min(v.x, mesh_min.x);
        mesh_min.y = std::min(v.y, mesh_min.y);
        mesh_min.z = std::min(v.z, mesh_min.z);

        mesh_max.x = std::max(v.x, mesh_max.x);
        mesh_max.y = std::max(v.y, mesh_max.y);
        mesh_max.z = std::max(v.z, mesh_max.z);
      }

      for (size_t i = 0; i < s.triangles.size(); ++i) {
        // flip orientation
        triangles3d.emplace_back(uint3{s.triangles[i].x + offset,
                                       s.triangles[i].z + offset,
                                       s.triangles[i].y + offset});
      }
      offset += s.vertices.size();
    }
  }

  write_ribbon_vtk("ribbon.vtk", vertices3d, triangles3d);

  // ...

  // build mesh
  // std::vector<double> vertices;
  // float3 mesh_min;
  // float3 mesh_max;
  // loadMesh(filename, vertices, mesh_min, mesh_max);

  // build query grid
  const int N = 8192;
  const double dx = (mesh_max.x - mesh_min.x)/N;
  const double dy = (mesh_max.y - mesh_min.y)/N;
  const double z = (mesh_max.z + mesh_min.z)/2;

  int padding = 1;
  std::vector<float3> query_points;
  for(int i=-padding; i<N+padding; i++) {
    for(int j=-padding; j<N+padding; j++) {
      query_points.push_back(make_float3(i*dx + mesh_min.x, j*dy + mesh_min.y, z));
    }
  }


  float3* d_query_points;
  CUDA_CHECK(cudaMalloc(&d_query_points, sizeof(float3)*query_points.size()));
  CUDA_CHECK(cudaMemcpy(d_query_points, query_points.data(), sizeof(float3)*query_points.size(), cudaMemcpyHostToDevice));

  float* d_query_distance;
  CUDA_CHECK(cudaMalloc(&d_query_distance, sizeof(float)*query_points.size()));
  float3* d_query_hit_points;
  CUDA_CHECK(cudaMalloc(&d_query_hit_points, sizeof(float3)*query_points.size()));
  int* d_query_triID;
  CUDA_CHECK(cudaMalloc(&d_query_triID, sizeof(int)*query_points.size()));

  //// compute signed distance field

  // pre-allocate result arrays
  std::vector<float> query_results_distance(query_points.size());

  computeSignedDistanceField(vertices3d, triangles3d, query_points.size(), d_query_points, d_query_distance, d_query_hit_points, d_query_triID);


  {
    Timer timer("Copy distance to host");
    CUDA_CHECK(cudaMemcpy(query_results_distance.data(), d_query_distance, sizeof(float)*query_points.size(), cudaMemcpyDeviceToHost));
  }

  // write SDF to VTK for visualization in Paraview
  // write_grid2d_vtk("output.vtk", make_float3(mesh_min.x-padding*dx, mesh_min.y-padding*dy, z),
  //                  dx, dy, z, N + 2*padding, N + 2*padding, query_results_distance);


  CUDA_CHECK(cudaFree(d_query_points));
  CUDA_CHECK(cudaFree(d_query_distance));
  CUDA_CHECK(cudaFree(d_query_hit_points));
  CUDA_CHECK(cudaFree(d_query_triID));

  return EXIT_SUCCESS;

}
