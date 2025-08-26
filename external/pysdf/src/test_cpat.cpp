#include <iostream>
#include <stdexcept>

#define TINYOBJLOADER_IMPLEMENTATION

#include <gequel/cpat/cpat.h>
#include <gequel/cpat/cpu/BruteForce.h>
#include <gequel/cpat/cpu/BVH2.h>
#include <gequel/cpat/gpu/BVH2.h>
#include <gequel/common/data_loader.h>

#include <cuda_runtime.h>

// void distance_field(const std::vector<float3>& mesh_vertices, const std::vector<int3>& mesh_indices, const std::vector<float3>& distance_field_pts, std::vector<float>& pts_distance, std::vector<int>& query_points_hit_triID, std::vector<float3>& closest_hit_point);

void test_tet() {
  std::vector<float3> mesh_vertices;
  std::vector<int3> mesh_indices;
  
  mesh_vertices.push_back(make_float3(0, 0, 0)); // bottom
  mesh_vertices.push_back(make_float3(0, 1, 0));
  mesh_vertices.push_back(make_float3(1, 0, 0));
  mesh_vertices.push_back(make_float3(0, 0, 0)); // front
  mesh_vertices.push_back(make_float3(1, 0, 0));
  mesh_vertices.push_back(make_float3(0, 0, 1));
  mesh_vertices.push_back(make_float3(0, 0, 0)); // left
  mesh_vertices.push_back(make_float3(0, 0, 1));
  mesh_vertices.push_back(make_float3(0, 1, 0));
  mesh_vertices.push_back(make_float3(1, 0, 0)); // top
  mesh_vertices.push_back(make_float3(0, 1, 0));
  mesh_vertices.push_back(make_float3(0, 0, 1));

  for(int i=0; i<4; i++) {
    mesh_indices.push_back(make_int3(3*i, 3*i+1, 3*i+2));
  }

  const int num_points = 1;

  std::vector<float3> query_points;
  std::vector<float> query_points_distance(num_points);
  std::vector<float3> query_points_hit(num_points);
  std::vector<gequel::vec3f> queryPoints(num_points);
  queryPoints[0].x = 1;
  queryPoints[0].y = 1;
  queryPoints[0].z = 1;

  float scaling = 0.001;
  float3 center = make_float3(-0.008, 0.131, 0.002);

  for(auto& qp : queryPoints) {
    query_points.push_back(make_float3(qp.x, qp.y, qp.z));
  }

  std::vector<int> query_points_hit_id(num_points);
  throw std::runtime_error("REFACTORING NOT DONE");
  // distance_field(mesh_vertices, mesh_indices, query_points, query_points_distance, query_points_hit_id, query_points_hit);

  std::ofstream closestPointsCSV("cpat-tet-closest-points.csv");
  std::ofstream queryPointsCSV("cpat-tet-query-points.csv");
  std::cout << "closest point file: cpat-closest-points.csv\n";
  std::cout << "query point file: cpat-query-points.csv\n";
  closestPointsCSV << "id, x, y, z\n";
  queryPointsCSV << "x, y, z\n";
  
  for (int i = 0; i < query_points.size(); i++) {
    queryPointsCSV << query_points[i].x << "," << query_points[i].y
                   << "," << query_points[i].z << std::endl;
    closestPointsCSV << query_points_hit_id[i] << ","
                     << query_points_hit[i].x << ","
                     << query_points_hit[i].y << ","
                     << query_points_hit[i].z << std::endl;
  }

}


int main() {

  test_tet();
  return 0;

  std::string filename = "/home/max/Downloads/bunny.obj";

  auto dataSet = gequel::createPoints_obj(filename);

  std::vector<float3> mesh_vertices_orig;
  std::vector<int3> mesh_indices_orig;

  for(auto &t : dataSet.first) {
    mesh_vertices_orig.push_back(make_float3(t.x, t.y, t.z));
  }

  for(auto &t : dataSet.second) {
    mesh_indices_orig.push_back(make_int3(t.x, t.y, t.z));
  }

  std::vector<float3> mesh_vertices;
  std::vector<int3> mesh_indices;
  for(int i=0; i<mesh_indices_orig.size(); i++) {
    if(mesh_indices_orig[i].x==0 ||
       mesh_indices_orig[i].y==0 ||
       mesh_indices_orig[i].z==0) {
      printf("0000000000000000000000\n");
    }
    mesh_vertices.push_back(mesh_vertices_orig[mesh_indices_orig[i].x]);
    mesh_vertices.push_back(mesh_vertices_orig[mesh_indices_orig[i].y]);
    mesh_vertices.push_back(mesh_vertices_orig[mesh_indices_orig[i].z]);
    mesh_indices.push_back(make_int3(3*i+0, 3*i+1, 3*i+2));
  }


  const int num_points = 100;
  std::vector<float3> query_points;
  std::vector<float> query_points_distance(num_points);
  std::vector<float3> query_points_hit(num_points);
  std::vector<gequel::vec3f> queryPoints = gequel::createPoints_uniform(num_points);

  float scaling = 0.001;
  float3 center = make_float3(-0.008, 0.131, 0.002);

  for(auto& qp : queryPoints) {
    query_points.push_back(make_float3(scaling*(qp.x-0.5)+center.x, scaling*(qp.y-0.5)+center.y, scaling*(qp.z-0.5)+center.z));
  }
  query_points[0] = make_float3(-0.008, 0.131, 0.002);

  std::vector<int> query_points_hit_id(num_points);
  
  throw std::runtime_error("REFACTORING NOT DONE");
  // distance_field(mesh_vertices, mesh_indices, query_points, query_points_distance, query_points_hit_id, query_points_hit);

  std::ofstream closestPointsCSV("cpat-closest-points.csv");
  std::ofstream queryPointsCSV("cpat-query-points.csv");
  std::cout << "closest point file: cpat-closest-points.csv\n";
  std::cout << "query point file: cpat-query-points.csv\n";
  closestPointsCSV << "id, x, y, z\n";
  queryPointsCSV << "x, y, z\n";
  
  for (int i = 0; i < query_points.size(); i++) {
    queryPointsCSV << query_points[i].x << "," << query_points[i].y
                   << "," << query_points[i].z << std::endl;
    closestPointsCSV << query_points_hit_id[i] << ","
                     << query_points_hit[i].x << ","
                     << query_points_hit[i].y << ","
                     << query_points_hit[i].z << std::endl;
  }
}

