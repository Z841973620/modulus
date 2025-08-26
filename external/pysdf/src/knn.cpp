#include <iostream>

// #include <gequel/
#include <gequel/knn/gpu/BVH2.h>

#include <cuda_runtime.h>
#include <shared_types.h>
#include <memory>
#include <knn.h>

using namespace gequel;

void findKnn(const std::vector<float3>& all_points, const std::vector<float3>& query_points,
             const int k,
             std::vector<std::vector<float3>>& result_points,
             float max_radius=-1) {

  result_points.resize(query_points.size());

  std::unique_ptr<KNNEngine> knn(new knn::gpu::BVH2EngineBT);

  std::vector<owl::vec3f> points(all_points.size());
  for(int i=0; i<all_points.size(); i++) {
    points[i].x = all_points[i].x;
    points[i].y = all_points[i].y;
    points[i].z = all_points[i].z;
  }

  std::unique_ptr<abstract::PointData> model(knn->createModel(points));

  vec3f* d_queryPoints;
  CUDA_CHECK(cudaMalloc(&d_queryPoints, sizeof(float3) * query_points.size()));
  CUDA_CHECK(cudaMemcpy(d_queryPoints, query_points.data(), sizeof(float3) * query_points.size(), cudaMemcpyHostToDevice));

  KNNResult* d_queryResults;
  std::vector<KNNResult> queryResults(query_points.size() * k);
  CUDA_CHECK(cudaMalloc(&d_queryResults, query_points.size() * k * sizeof(KNNResult)));

  if (max_radius < 0) {
    knn->findKNNs(d_queryResults, query_points.size(), model.get(),
                  d_queryPoints, k);
  }
  else {
    knn->findKNNs(d_queryResults, query_points.size(), model.get(), d_queryPoints, k, max_radius);
  }

  CUDA_CHECK(cudaMemcpy(queryResults.data(), d_queryResults, query_points.size() * k * sizeof(KNNResult), cudaMemcpyDeviceToHost));

  for(int i=0; i<query_points.size(); i++) {
    for(int j=0; j<k; j++) {

      // if no points is possible, it leaves a -1 id
      if(queryResults[i * k + j].pointID >= 0) {
        result_points[i].push_back(all_points[queryResults[i * k + j].pointID]);
      }
    }
  }

  CUDA_CHECK(cudaFree(d_queryPoints));
  CUDA_CHECK(cudaFree(d_queryResults));
}

#ifdef __cplusplus
extern "C"
#endif
void findKnn(int num_all_points, double* all_points,
             int num_query_points, double* query_points,
             int k,
             double* result_points_all,
             int* num_result_points_per_query
             ) {


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
  findKnn(vec_all_points, vec_query_points, k, result_points, -1);

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
