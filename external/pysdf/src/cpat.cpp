#include "owl/common/math/vec.h"
#include <iostream>

#include <gequel/cpat/cpat.h>
#include <gequel/cpat/cpu/BruteForce.h>
#include <gequel/cpat/cpu/BVH2.h>
#include <gequel/cpat/gpu/BVH2.h>

#include <cuda_runtime.h>
#include <shared_types.h>
#include <memory>

#include <chrono>

#include<nvtx3/nvToolsExt.h>

#include "timing.h"

using namespace gequel;

void cpatResultsToArrays(const size_t num_query_points, CPATResult* d_query_results,
                         float* d_query_points_distance, int* d_query_points_hit_triID, float3* d_query_points_hit);


void cpatDistanceField(const std::vector<float3>& mesh_vertices, const std::vector<uint3>& mesh_indices,
                       const size_t num_query_points, float3* d_query_points,
                       float* d_query_points_distance, int* d_query_points_hit_triID, float3* d_query_points_hit) {

  Timer timer("cpatDistanceField");

  std::cout << "NUM Triangles: " << mesh_indices.size() << "\n";

  std::vector<std::unique_ptr<CPATEngine>> engines;
  engines.emplace_back(new cpat::gpu::BVH2EngineBT);

  std::unique_ptr<CPATEngine> cpat(new cpat::gpu::BVH2EngineBT);

  std::vector<owl::vec3f> vertices(mesh_vertices.size());
  std::vector<owl::vec3i> faces(mesh_indices.size());
  for(int i=0;i<mesh_vertices.size();i++) {
    vertices[i].x = mesh_vertices[i].x;
    vertices[i].y = mesh_vertices[i].y;
    vertices[i].z = mesh_vertices[i].z;
  }
  for(int i=0;i<mesh_indices.size();i++) {
    faces[i].x = mesh_indices[i].x;
    faces[i].y = mesh_indices[i].y;
    faces[i].z = mesh_indices[i].z;
  }

  Timer* timer2 = new Timer("Build CPAT Model (build bvh)");
  std::unique_ptr<abstract::PointData> model(cpat->createModel(vertices, faces));
  delete timer2;

  vec3f* d_query_points_vec3f = (vec3f*)d_query_points;
  CPATResult *d_query_results;
  CUDA_CHECK(
      cudaMalloc(&d_query_results, num_query_points * sizeof(CPATResult)));

  cpat->findCPATs(d_query_results, num_query_points, model.get(),
                  d_query_points_vec3f);

  cpatResultsToArrays(num_query_points, d_query_results, d_query_points_distance, d_query_points_hit_triID, d_query_points_hit);

  cudaFree(d_query_results);
}
