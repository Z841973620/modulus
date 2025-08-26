#include <iostream>

#include <cuda_runtime.h>
#include <shared_types.h>
#include <memory>
#include <limits>
#include <prq.h>

void findKnn(const std::vector<float3>& all_points, const std::vector<float3>& query_points,
             const int k,
             std::vector<std::vector<float3>>& result_points,
             float max_radius=-1.0);


void point_radius_query(const std::vector<float3>& all_points, const std::vector<float3>& query_points, const float radius, std::vector<std::vector<float3>>& points_within_radius, int k) {
  
  findKnn(all_points, query_points, k, points_within_radius, radius);
}

#ifdef __cplusplus
extern "C"
#endif
void pointRadiusQuery(int num_all_points, double* all_points,
                      int num_query_points, double* query_points,
                      double max_radius,
                      int k,
                      double* result_points_all,
                      int* num_result_points_per_query) {

  std::vector<float3> vec_all_points(num_all_points);
  for(int i=0; i<num_all_points; i++) {
    vec_all_points[i].x = all_points[3*i+0];
    vec_all_points[i].y = all_points[3*i+1];
    vec_all_points[i].z = all_points[3*i+2];
  }

  std::vector<float3> vec_query_points(num_query_points);
  for(int i=0; i<num_query_points; i++) {
    vec_query_points[i].x = query_points[3*i + 0];
    vec_query_points[i].y = query_points[3*i + 1];
    vec_query_points[i].z = query_points[3*i + 2];
  }

  std::vector<std::vector<float3>> result_points;
  point_radius_query(vec_all_points, vec_query_points, max_radius, result_points, k);

  int num_results_total = 0;
  for(int i=0; i<num_query_points; i++) {
    num_result_points_per_query[i] = result_points[i].size();
    for(int j=0; j<result_points[i].size(); j++) {
      result_points_all[3*num_results_total + 0] = result_points[i][j].x;
      result_points_all[3*num_results_total + 1] = result_points[i][j].y;
      result_points_all[3*num_results_total + 2] = result_points[i][j].z;
      num_results_total++;
    }
  }
}

#ifdef __cplusplus
extern "C"
#endif
void freeResultPoints(double* results_points_all) {
  free(results_points_all);
}
