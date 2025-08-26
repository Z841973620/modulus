#include <iostream>

#include <gequel/cpat/cpat.h>
#include <gequel/cpat/cpu/BruteForce.h>
#include <gequel/cpat/cpu/BVH2.h>
#include <gequel/cpat/gpu/BVH2.h>

#include <cuda_runtime.h>
#include <shared_types.h>
#include <memory>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <timing.h>
using namespace gequel;

void cpatComputeSignedFieldWithNormals(const size_t num_queries,
                                       float3* d_query_points,
                                       float* d_query_points_distance,
                                       float3* d_query_points_hit,
                                       int* d_query_points_triID,
                                       float3* d_vertices,
                                       int3* d_triangles) {

  Timer timer("cpatComputeSignedFieldWithNormals");
  auto counter = thrust::make_counting_iterator(0);
  thrust::for_each(counter, counter + num_queries,
                   [=] __device__(int idx) {
                     auto hit_id = d_query_points_triID[idx];

                     float3 v0 = d_vertices[d_triangles[hit_id].x];
                     float3 v1 = d_vertices[d_triangles[hit_id].y];
                     float3 v2 = d_vertices[d_triangles[hit_id].z];

                     float3 normal = cross((v1-v0), (v2-v0));

                     float3& qp = d_query_points[idx];

                     if(dot(d_query_points_hit[idx] - qp, normal) > 0) {
                       d_query_points_distance[idx] *= -1;
                     }
                   });

}


// turn CPATResult into arrays
void cpatResultsToArrays(const size_t num_query_points, CPATResult* d_query_results,
                         float* d_query_points_distance, int* d_query_points_hit_triID, float3* d_query_points_hit) {

  Timer timer("cpatResultsToArrays");
  auto counter = thrust::make_counting_iterator(0);
  thrust::for_each(counter, counter + num_query_points,
                   [=] __device__(int idx) {
                     d_query_points_distance[idx] = d_query_results[idx].dist;
                     d_query_points_hit_triID[idx] = d_query_results[idx].triID;
                     d_query_points_hit[idx] = (float3)d_query_results[idx].point;
                   });
}


void applySignedField(const size_t num_query_points, float* d_query_distance, int* d_signed_field_results) {
  auto counter = thrust::make_counting_iterator(0);
  thrust::for_each(counter, counter + num_query_points,
                   [=] __device__(int idx) {
                     if(d_signed_field_results[idx] == CELL_TYPE_INSIDE) {
                       d_query_distance[idx] *= -1;
                     }
                   });
}
