#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include <cuda_runtime.h>

#include <sdf_cpp.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
using std::vector;

#define CUDA_CHECK(A)                                     \
  do {                                                    \
    cudaError_t err = A;                                  \
    if (err != cudaSuccess) {                             \
      throw std::runtime_error(cudaGetErrorString(err));  \
    }                                                     \
  } while (false)

//! Create regular 3D grid in VTK with "grid_values" as cell values.
void write_grid3d_vtk(const std::string& filename, float3 origin, float dx, float dy, float dz, int Nx, int Ny, int Nz,
                      const std::vector<float>& grid_values) {
  std::ofstream outFile(filename);

  outFile << "# vtk DataFile Version 2.0\n";
  outFile << "Grid\n";
  outFile << "ASCII\n";
  outFile << "DATASET STRUCTURED_GRID\n";
  outFile << "DIMENSIONS " << Nx << " " << Ny << " " << Nz << std::endl;
  outFile << "POINTS " << (Nx)*(Ny)*(Nz) << " float\n";
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        outFile << origin.x + i * dx << " " << origin.y + j * dy << " "
                << origin.z + k * dz << "\n";
      }
    }
  }

  outFile << "POINT_DATA " << grid_values.size() << "\n";
  outFile << "SCALARS scalars double 1\n";
  outFile << "LOOKUP_TABLE default\n";
  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        outFile << grid_values[Ny*Nz*i + Nz*j + k] << "\n";
      }
    }
  }
}


// Reads mesh 'filename' and returns vertices that are grouped per triangle.
bool loadMesh(const std::string& filename, vector<float3>& vertices, vector<uint3>& triangles, float3& mesh_min, float3& mesh_max) {

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

        vector<int> idx3;

        for (size_t v = 0; v < fv; v++) {
          // access to vertex
          tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

          if (idx.vertex_index >= 0) {

            idx3.push_back(idx.vertex_index);

            double vx = attrib.vertices[3 * idx.vertex_index + 0];
            double vy = attrib.vertices[3 * idx.vertex_index + 1];
            double vz = attrib.vertices[3 * idx.vertex_index + 2];
            
            // update mesh bounds
            mesh_min.x = std::min((float)vx, mesh_min.x);
            mesh_min.y = std::min((float)vy, mesh_min.y);
            mesh_min.z = std::min((float)vz, mesh_min.z);

            mesh_max.x = std::max((float)vx, mesh_max.x);
            mesh_max.y = std::max((float)vy, mesh_max.y);
            mesh_max.z = std::max((float)vz, mesh_max.z);

          }
        }

        triangles.push_back(make_uint3(idx3[0], idx3[1], idx3[2]));
        index_offset += fv;

      }
    }
  }

  for(int i=0; i<attrib.vertices.size()/3; i++) {
    vertices.push_back(make_float3(attrib.vertices[3*i+0],
                                   attrib.vertices[3*i+1],
                                   attrib.vertices[3*i+2]));
  }

  return true;
}


int main(int argc, char** argv) {
  std::cout << "HELLO\n";
  std::string filename;
  std::string filename2;

  if(argc == 1) {
    printf("Usage: sdf [-h] [mesh1.obj] [mesh2.obj]");
    return EXIT_FAILURE;
  }
  if(argc > 1) {
    if(std::string(argv[1]) == std::string("-h")) {
      printf("Usage: sdf [-h] [mesh1.obj] [mesh2.obj]");
      return EXIT_FAILURE;
    }
    else {
      filename = std::string(argv[1]);
      filename2 = std::string(argv[2]);
    }
  }
  
  // load mesh
  std::vector<float3> vertices;
  std::vector<uint3> triangles;
  float3 mesh_min;
  float3 mesh_max;
  loadMesh(filename, vertices, triangles, mesh_min, mesh_max);

  // load mesh
  std::vector<float3> vertices2;
  std::vector<uint3> triangles2;
  float3 mesh_min2;
  float3 mesh_max2;
  loadMesh(filename2, vertices2, triangles2, mesh_min2, mesh_max2);

  mesh_min.x = std::min(mesh_min.x, mesh_min2.x);
  mesh_min.y = std::min(mesh_min.y, mesh_min2.y);
  mesh_min.z = std::min(mesh_min.z, mesh_min2.z);

  mesh_max.x = std::max(mesh_max.x, mesh_max2.x);
  mesh_max.y = std::max(mesh_max.y, mesh_max2.y);
  mesh_max.z = std::max(mesh_max.z, mesh_max2.z);

  // merge meshes
  const size_t num_vertices1 = vertices.size();
  const size_t num_triangles1 = triangles.size();
  const size_t num_triangles2 = triangles2.size();
  vertices.insert(vertices.end(), vertices2.begin(), vertices2.end());
  for(int i=0; i<num_triangles2; i++) {
    auto i0 = triangles2[i].x;
    auto i1 = triangles2[i].y;
    auto i2 = triangles2[i].z;
    triangles.push_back(make_uint3(i0+num_vertices1, i1+num_vertices1, i2+num_vertices1));
  }

  // build query grid
  const int N = 100;
  const double dx = (mesh_max.x - mesh_min.x)/N;
  const double dy = (mesh_max.y - mesh_min.y)/N;
  const double dz = (mesh_max.z - mesh_min.z)/N;

  int padding = 20;
  std::vector<float3> query_points;
  for(int i=-padding; i<N+padding; i++) {
    for(int j=-padding; j<N+padding; j++) {
      for (int k=-padding; k < N+padding; k++) {
        query_points.push_back(make_float3(i*dx + mesh_min.x,
                                           j*dy + mesh_min.y,
                                           k*dz + mesh_min.z));
      }
    }
  }

  //// compute signed distance field

  float3* d_query_points;
  CUDA_CHECK(cudaMalloc(&d_query_points, sizeof(float3)*query_points.size()));
  CUDA_CHECK(cudaMemcpy(d_query_points, query_points.data(), sizeof(float3)*query_points.size(), cudaMemcpyHostToDevice));

  float* d_query_distance;
  CUDA_CHECK(cudaMalloc(&d_query_distance, sizeof(float)*query_points.size()));
  float3* d_query_hit_points;
  CUDA_CHECK(cudaMalloc(&d_query_hit_points, sizeof(float3)*query_points.size()));
  int* d_query_triID;
  CUDA_CHECK(cudaMalloc(&d_query_triID, sizeof(int)*query_points.size()));


  // pre-allocate result arrays
  std::vector<float> query_results_distance(query_points.size());
  std::vector<float> query_results_hit(query_points.size());
  vector<int> query_triID(query_points.size());

  // compute signed distance field
  computeSignedDistanceField(vertices, triangles,
                             query_points.size(), d_query_points,
                             d_query_distance, d_query_hit_points, d_query_triID,
                             SIGNED_FIELD_TECHNIQUE_INTERSECTION_COUNT);

  CUDA_CHECK(cudaMemcpy(query_results_distance.data(), d_query_distance, sizeof(float)*query_points.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(query_triID.data(), d_query_triID, sizeof(int)*query_points.size(), cudaMemcpyDeviceToHost));

  vector<float> object_id(query_points.size());
  vector<float> query_triID_float(query_points.size());
  // determine object based on triID
  for(int i=0; i<query_triID.size(); i++) {
    if(query_triID[i] < num_triangles1) {
      object_id[i] = 0.0;
    }
    else {
      object_id[i] = 1.0;
    }
    query_triID_float[i] = query_triID[i];
  }

  // write SDF to VTK for visualization in Paraview
  write_grid3d_vtk("multi_sdf.vtk", make_float3(mesh_min.x-padding*dx, mesh_min.y-padding*dy, mesh_min.z-padding*dz),
                   dx, dy, dz, N + 2*padding, N + 2*padding, N + 2*padding, query_results_distance);

  write_grid3d_vtk("multi_objID.vtk", make_float3(mesh_min.x-padding*dx, mesh_min.y-padding*dy, mesh_min.z-padding*dz),
                   dx, dy, dz, N + 2*padding, N + 2*padding, N + 2*padding, object_id);

  // write_grid3d_vtk("multi_indexID.vtk", make_float3(mesh_min.x-padding*dx, mesh_min.y-padding*dy, mesh_min.z-padding*dz),
  //                  dx, dy, dz, N + 2*padding, N + 2*padding, N + 2*padding, query_triID_float);

  CUDA_CHECK(cudaFree(d_query_points));
  CUDA_CHECK(cudaFree(d_query_distance));
  CUDA_CHECK(cudaFree(d_query_hit_points));
  CUDA_CHECK(cudaFree(d_query_triID));


  return EXIT_SUCCESS;

}
