/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>
#include <stdexcept>
#include <streambuf>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <iostream>
#include <iomanip>
#include <limits>
#include <algorithm>


#include <string>
#include <vector>

#include <optix.h>
#include <cuda_runtime.h>


#include <shared_types.h>

#include <sdf.h>

#include <random>

#include <cassert>

#include <chrono>

// includes auto-generated resource header
#include <grid.h>

#include <nvtx3/nvToolsExt.h>

#include "helper_math.h"
#include "timing.h"

using std::vector;
using std::printf;

#define OPTIX_CHECK(A)                                                         \
  do {                                                                         \
    OptixResult err = A;                                                       \
    if (err != OPTIX_SUCCESS) {                                                \
      throw std::runtime_error(optixGetErrorString(err));                      \
    }                                                                          \
  } while (false)

static void optix_log_callback(unsigned int level, const char *tag,
                               const char *message, void *) {
  std::clog << "[OptiX][" << std::setw(2) << level << "][" << std::setw(12)
            << tag << "]: " << message << "\n";
}

template <typename T> struct SbtRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

// Utility function to round 'x' up to the next multiple of y
// Ex: 
// roundUp(2,3) -> 3
// roundUp(8,3) -> 9
// roundUp(10,3) -> 12
// roundUp(13,3) -> 12
// roundUp(13,3) -> 15
// roundUp(13,8) -> 16
// roundUp(17,8) -> 24
template <typename IntegerType>
__device__ __host__ IntegerType roundUp(IntegerType x, IntegerType y) {
  return ((x + y - 1) / y) * y;
}

struct RayGenData {
  int counter;
};
struct MissData {
  int counter;
};

struct HitGroupData {
  int counter;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

std::vector<float3> makeFibonacciSphere(const int NPOINTS) {
  std::vector<float3> sphere_pts(NPOINTS);
  float phi = 2.39996322973; // golden ratio
  for (int i = 0; i < NPOINTS; i++) {
    float y = 1.0 - ((float)(i) / (float)(NPOINTS - 1)) * 2.0;
    float radius = std::sqrt(1.0 - y * y);
    float theta = phi * (float)(i);

    float x = std::cos(theta) * radius;
    float z = std::sin(theta) * radius;
    float norm = std::sqrt(x*x + y*y + z*z);
    sphere_pts[i] = make_float3(x/norm, y/norm, z/norm);
  }

  return sphere_pts;
}

void cleanupContext(OptixDeviceContext context) {

  OPTIX_CHECK(optixDeviceContextDestroy(context));

}

OptixDeviceContext initContext() {
  // 
  OPTIX_CHECK(optixInit());
  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &optix_log_callback;
  options.logCallbackLevel = 4;
  OptixDeviceContext optix_context;
  // 
  OPTIX_CHECK(optixDeviceContextCreate(0, &options, &optix_context));
  return optix_context;
} 

void createModule(OptixDeviceContext optix_context,
                  OptixModule& module, OptixPipelineCompileOptions& pipeline_compile_options, unsigned int num_payload
                  ) {
  std::string ptx(reinterpret_cast<const char*>(grid_ptx_storage));

  OptixModuleCompileOptions module_compile_options = {};
  module_compile_options.maxRegisterCount =
      OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  pipeline_compile_options.usesMotionBlur = false;
  pipeline_compile_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_compile_options.numPayloadValues = num_payload;
  pipeline_compile_options.numAttributeValues = 2;
  pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  // 
  OPTIX_CHECK(optixModuleCreateFromPTX(optix_context, &module_compile_options,
                                       &pipeline_compile_options, ptx.c_str(),
                                       ptx.size(), NULL, NULL, &module));
  
}

OptixProgramGroup createProgramGroup(OptixDeviceContext optix_context,
                                     OptixModule module,
                                     OptixProgramGroupKind kind,
                                     std::string name,
                                     bool closest_hit) {
  OptixProgramGroupOptions program_group_options;

  OptixProgramGroupDesc prog_group_desc = {}; //
  prog_group_desc.kind = kind; // e.g. OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  switch(kind) {
  case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
    prog_group_desc.raygen.module = module;
    prog_group_desc.raygen.entryFunctionName = name.c_str();
    break;
  case OPTIX_PROGRAM_GROUP_KIND_MISS:
    if(name != "") {
      prog_group_desc.miss.module = module;
      prog_group_desc.miss.entryFunctionName = name.c_str();
    }
    break;
  case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:

    if(closest_hit) {
      prog_group_desc.hitgroup.moduleCH = module;
      prog_group_desc.hitgroup.entryFunctionNameCH = name.c_str();
      
    } else {
      prog_group_desc.hitgroup.moduleAH = module;
      prog_group_desc.hitgroup.entryFunctionNameAH = name.c_str();
    }
    break;
  default:
    throw std::runtime_error("UNSUPPORTED PROGRAM GROUP TYPE");
  }
  OptixProgramGroup prog_group;
  
  OPTIX_CHECK(optixProgramGroupCreate(optix_context, &prog_group_desc,
                                      1, // num program groups
                                      &program_group_options, nullptr, nullptr,
                                      &prog_group));

  return prog_group;
}

OptixPipeline linkPipeLine(OptixDeviceContext optix_context, const OptixPipelineCompileOptions& pipeline_compile_options,
                           const std::vector<OptixProgramGroup>& program_groups) {
  OptixPipeline pipeline;
  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = 5;
  pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  OPTIX_CHECK(optixPipelineCreate(optix_context, &pipeline_compile_options, &pipeline_link_options,
                                  program_groups.data(), program_groups.size(),
                                  nullptr, nullptr, &pipeline));
  return pipeline; 
}
void buildSBT(OptixShaderBindingTable& sbt,
              OptixProgramGroup raygen_prog_group,
              OptixProgramGroup miss_prog_group,
              OptixProgramGroup hitgroup_prog_group) {


  // raygen
  void *raygenRecord;
  size_t raygenRecordSize = sizeof(RayGenSbtRecord);
  CUDA_CHECK(cudaMalloc((void **)&raygenRecord, raygenRecordSize));
  RayGenSbtRecord rgSBT;

  OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rgSBT));
  rgSBT.data = {0};
  CUDA_CHECK(cudaMemcpy((void *)raygenRecord, &rgSBT, raygenRecordSize,
                        cudaMemcpyHostToDevice));
  sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygenRecord);

  // miss
  void *missSbtRecord;
  size_t missSbtRecordSize = sizeof(MissSbtRecord);
  CUDA_CHECK(cudaMalloc((void **)&missSbtRecord, missSbtRecordSize));
  MissSbtRecord msSBT;
  OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &msSBT));
  msSBT.data = {0};
  CUDA_CHECK(cudaMemcpy(missSbtRecord, &msSBT, missSbtRecordSize,
                        cudaMemcpyHostToDevice));

  sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(missSbtRecord);
  sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
  sbt.missRecordCount = 1;

  // hit
  void *hitgroupSbtRecord;
  size_t hitgroupSbtRecordSize = sizeof(HitGroupSbtRecord);
  CUDA_CHECK(cudaMalloc((void **)&hitgroupSbtRecord, hitgroupSbtRecordSize));
  HitGroupSbtRecord hgSBT;
  // 
  OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hgSBT));
  hgSBT.data = {0};
  CUDA_CHECK(cudaMemcpy(hitgroupSbtRecord, &hgSBT, hitgroupSbtRecordSize,
                        cudaMemcpyHostToDevice));
  
  sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroupSbtRecord);
  sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
  sbt.hitgroupRecordCount = 1;

}

void buildAccelStructure(OptixDeviceContext optix_context, int num_vertices, float3* d_vertices, int num_triangles, uint3* d_indices, OptixTraversableHandle& gas_handle) {
  Timer timer("buildAccelStructure");
  // steps
  // 1. put vertices on device
  // 2. prepare "triangle" input to acceleration structure
  // 3. prepare space for acceleration structure
  // 4. build acceleration structure

  // 1. copy vertices
  
  // 2. prepare triangle input

  // REQUIRE_SINGLE_ANYHIT_CALL is critical, because we use anyhit to count
  // intersections, and without this flag, it is possible to count the same
  // triangle twice, depending on the BVH construction.
  const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
  OptixBuildInput triangle_input = {};
  triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.numVertices = num_vertices;
  triangle_input.triangleArray.vertexBuffers =
    reinterpret_cast<CUdeviceptr *>(&d_vertices);
  triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(d_indices);
  triangle_input.triangleArray.numIndexTriplets = num_triangles;
  triangle_input.triangleArray.flags = triangle_input_flags;
  triangle_input.triangleArray.numSbtRecords = 1;

  // 3. prepare space
  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context, &accel_options,
                                           &triangle_input,
                                           1, // Number of build input
                                           &gas_buffer_sizes));

  void *d_temp_buffer_gas;
  CUDA_CHECK(cudaMalloc(&d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes));

  // non-compacted output
  void *d_buffer_temp_output_gas_and_compacted_size;
  size_t compactedSizeOffset =
    roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
  CUDA_CHECK(cudaMalloc(&d_buffer_temp_output_gas_and_compacted_size,
                        compactedSizeOffset + 8));

  OptixAccelEmitDesc emitProperty = {};
  emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitProperty.result =
    (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size +
                  compactedSizeOffset);
  {
    Timer timer("optixAccelBuild");
    OPTIX_CHECK(optixAccelBuild(optix_context,
                              0, // CUDA stream
                              &accel_options, &triangle_input,
                              1, // num build inputs
                              reinterpret_cast<CUdeviceptr>(d_temp_buffer_gas),
                              gas_buffer_sizes.tempSizeInBytes,
                              reinterpret_cast<CUdeviceptr>(
                                                            d_buffer_temp_output_gas_and_compacted_size),
                              gas_buffer_sizes.outputSizeInBytes, &gas_handle,
                              &emitProperty, // emitted property list
                              1              // num emitted properties
                              ));
  }
  // CUDA_CHECK(cudaFree(d_temp_buffer_gas));
  // CUDA_CHECK(cudaFree(d_buffer_temp_output_gas_and_compacted_size));
}

void computeInsideOutsidePoints(const std::vector<float3>& mesh_vertices,
                                const std::vector<uint3>& mesh_indices,
                                const size_t num_query_points,
                                float3* d_query_points,
                                int* d_signed_field_results,
                                const int NUM_SPHERE_PTS = 101,
                                const SIGNED_FIELD_TECHNIQUE signed_technique = SIGNED_FIELD_TECHNIQUE_INTERSECTION_COUNT
                                ) {


  ////////////////////////////////////////////////////////////
  // prepare optix programs
  auto optix_context = initContext();
  OptixTraversableHandle gas_handle;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  float3* d_mesh_vertices;
  uint3* d_mesh_indices;

  cudaMalloc((void**)&d_mesh_vertices, sizeof(float3)*mesh_vertices.size());
  cudaMalloc((void**)&d_mesh_indices, sizeof(uint3)*mesh_indices.size());
  cudaMemcpy(d_mesh_vertices, mesh_vertices.data(), sizeof(float3)*mesh_vertices.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mesh_indices, mesh_indices.data(), sizeof(uint3)*mesh_indices.size(), cudaMemcpyHostToDevice);

  buildAccelStructure(optix_context, mesh_vertices.size(), d_mesh_vertices, mesh_indices.size(), d_mesh_indices, gas_handle);

  OptixModule module;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  createModule(optix_context, module, pipeline_compile_options, 3);
  
  OptixProgramGroup grid_raygen_prog_group = createProgramGroup(optix_context, module,
                                                                OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                                                                "__raygen__signed_field_intersection", false);
  OptixProgramGroup grid_miss_prog_group = createProgramGroup(optix_context, module, OPTIX_PROGRAM_GROUP_KIND_MISS, "__miss__signed_field", false);
  OptixProgramGroup grid_hitgroup_prog_group = createProgramGroup(optix_context, module, OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                                                                  "__anyhit__signed_field",
                                                                  false);

  OptixPipeline pipeline =
    linkPipeLine(optix_context, pipeline_compile_options, {grid_raygen_prog_group, grid_hitgroup_prog_group});

  OptixShaderBindingTable sbt = {};
  buildSBT(sbt, grid_raygen_prog_group, grid_miss_prog_group, grid_hitgroup_prog_group);

  ////////////////////////////////////////////////////////////
  // Launch Optix

  // setup input/output parameter
  Params params;

  params.d_mesh_vertices = d_mesh_vertices;
  params.signed_technique = signed_technique;

  auto sphere_pts = makeFibonacciSphere(NUM_SPHERE_PTS);

  cudaMalloc((void**)&params.d_sphere_pts, NUM_SPHERE_PTS*sizeof(float3));
  cudaMemcpy(params.d_sphere_pts, sphere_pts.data(), NUM_SPHERE_PTS*sizeof(float3), cudaMemcpyHostToDevice);
  params.num_sphere_pts = NUM_SPHERE_PTS;

  params.signed_field_results = (uint*)d_signed_field_results;
  params.num_mesh_points = mesh_vertices.size();

  params.signed_field_pts = d_query_points;
  params.handle = gas_handle;
  // copy to device
  Params* d_params;
  CUDA_CHECK(cudaMalloc(&d_params, sizeof(Params)));
  CUDA_CHECK(cudaMemcpy(d_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
  
  auto time0 = std::chrono::high_resolution_clock::now();
  OPTIX_CHECK(optixLaunch(pipeline, stream,
              reinterpret_cast<CUdeviceptr>(d_params),
                          sizeof(Params), &sbt, num_query_points, 1, 1));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto time0_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(time0_end - time0);
  std::cout << "Spherical Inside/Outside took: " << time_span.count() << "\n";  

  CUDA_CHECK(cudaFree(params.d_sphere_pts));
  CUDA_CHECK(cudaFree(params.d_mesh_vertices));
  CUDA_CHECK(cudaFree(d_params));
  CUDA_CHECK(cudaFree(d_mesh_indices));

  optixModuleDestroy(module);
  optixPipelineDestroy(pipeline);
  optixProgramGroupDestroy(grid_raygen_prog_group);
  optixProgramGroupDestroy(grid_miss_prog_group);
  optixProgramGroupDestroy(grid_hitgroup_prog_group);
  
  cleanupContext(optix_context);
  cudaStreamDestroy(stream);

}

void cpatDistanceField(const std::vector<float3>& mesh_vertices, const std::vector<uint3>& mesh_indices,
                       const size_t num_query_points, float3* d_query_points,
                       float* d_query_points_distance, int* d_query_points_hit_triID, float3* d_query_points_hit);

void cpatSignedDistanceField(const std::vector<float3>& mesh_vertices, const std::vector<uint3>& mesh_indices,
                             const size_t num_query_points, float3* d_query_points,
                             float* d_query_points_signed_distance, int* d_query_points_hit_triID, float3* d_query_points_hit);

void cpatComputeSignedFieldWithNormals(const size_t num_queries,
                                       float3* d_query_points,
                                       float* d_query_points_distance,
                                       float3* d_query_points_hit,
                                       int* d_query_points_triID,
                                       float3* d_vertices,
                                       int3* d_triangles);


void computeDistanceField(const std::vector<float3>& mesh_vertices,
                          const std::vector<uint3>& mesh_indices,
                          const size_t num_query_pts,
                          float3* d_query_pts,
                          float* d_query_distance,
                          float3* d_query_hit_point,
                          int* d_query_triID
                          ) {

  Timer timer("computeDistanceField");

  cpatDistanceField(mesh_vertices, mesh_indices, num_query_pts, d_query_pts, d_query_distance, d_query_triID, d_query_hit_point);

}

void applySignedField(const size_t num_query_points, float *d_query_distance,
                      int *d_signed_field_results);

void computeSignedDistanceField(const std::vector<float3>& mesh_vertices,
                                const std::vector<uint3>& mesh_indices,
                                const size_t num_query_pts,
                                float3* d_query_pts,
                                float* d_query_distance,
                                float3* d_query_hit_point,
                                int* d_query_triID,
                                const SIGNED_FIELD_TECHNIQUE signed_field_method
                                ) {

  Timer timer("computeSignedDistanceField");

  computeDistanceField(mesh_vertices, mesh_indices, num_query_pts, d_query_pts, d_query_distance, d_query_hit_point, d_query_triID);

  if(signed_field_method == SIGNED_FIELD_TECHNIQUE_NORMAL_ALIGNMENT) {
    float3* d_mesh_vertices;
    int3* d_triangles;
    cudaMalloc(&d_mesh_vertices, sizeof(float3)*mesh_vertices.size());
    cudaMalloc(&d_triangles, sizeof(int3)*mesh_indices.size());
    cudaMemcpy(d_mesh_vertices, mesh_vertices.data(), sizeof(float3)*mesh_vertices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, mesh_indices.data(), sizeof(int3)*mesh_indices.size(), cudaMemcpyHostToDevice);

    cpatComputeSignedFieldWithNormals(num_query_pts, d_query_pts, d_query_distance, d_query_hit_point, d_query_triID, d_mesh_vertices, d_triangles);
    cudaFree(d_mesh_vertices);
    cudaFree(d_triangles);
  } else if (signed_field_method == SIGNED_FIELD_TECHNIQUE_INTERSECTION_COUNT) {
    Timer timer("computeSignedField");
    int* d_signed_field_results;
    cudaMalloc(&d_signed_field_results, sizeof(int)*num_query_pts);
    computeInsideOutsidePoints(mesh_vertices, mesh_indices, num_query_pts, d_query_pts, d_signed_field_results);
    applySignedField(num_query_pts, d_query_distance, d_signed_field_results);
    cudaFree(d_signed_field_results);
  }

}




void signedDistanceField(int num_vertices, double* mesh_vertices,
                         int num_points, double* input_points,
                         double* points_signed_distance,
                         double* closest_hit_point) {

  vector<float3> vec_mesh_vertices(num_vertices);

  for (int i = 0; i < num_vertices; i++) {
    vec_mesh_vertices[i] =
        make_float3(mesh_vertices[3 * i + 0], mesh_vertices[3 * i + 1],
                    mesh_vertices[3 * i + 2]);
  }

  vector<float3> vec_input_points(num_points);
  for(int i=0; i<num_points; i++) {
    vec_input_points[i] = make_float3(input_points[3*i+0],
                                      input_points[3*i+1],
                                      input_points[3*i+2]);

  }

  ////////////////////////////////////////////////////////////
  // compute distance field on outside points
  std::vector<float> vec_points_distance(vec_input_points.size());
  std::vector<float3> vec_closest_hit_point(vec_input_points.size());

  // currently unused, however interface currently requires it.
  vector<float3> sphere_pts;

  auto time0 = std::chrono::high_resolution_clock::now();

  float3* d_query_points;
  CUDA_CHECK(cudaMalloc(&d_query_points, sizeof(float3)*vec_input_points.size()));
  CUDA_CHECK(cudaMemcpy(d_query_points, vec_input_points.data(), sizeof(float3)*vec_input_points.size(), cudaMemcpyHostToDevice));

  float* d_query_distance;
  CUDA_CHECK(cudaMalloc(&d_query_distance, sizeof(float)*vec_input_points.size()));
  float3* d_query_hit_points;
  CUDA_CHECK(cudaMalloc(&d_query_hit_points, sizeof(float3)*vec_input_points.size()));
  int* d_query_triID;
  CUDA_CHECK(cudaMalloc(&d_query_triID, sizeof(int)*vec_input_points.size()));

  // assumes triangles vertices are duplicated in the mesh.
  vector<uint3> vec_mesh_indices;
  for(int i=0; i<vec_mesh_vertices.size()/3; i++) {
    vec_mesh_indices.push_back(make_uint3(3*i, 3*i+1, 3*i+2));
  }

  computeSignedDistanceField(vec_mesh_vertices, vec_mesh_indices, vec_input_points.size(),
                             d_query_points, d_query_distance, d_query_hit_points, d_query_triID);

  ////////////////////////////////////////////////////////////
  // prepare output
  CUDA_CHECK(cudaMemcpy(vec_closest_hit_point.data(), d_query_hit_points, sizeof(float3)*vec_input_points.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(vec_points_distance.data(), d_query_distance, sizeof(float)*vec_input_points.size(), cudaMemcpyDeviceToHost));

  for(int i=0; i<num_points; i++) {
    points_signed_distance[i] = vec_points_distance[i];
    closest_hit_point[3*i + 0] = vec_closest_hit_point[i].x;
    closest_hit_point[3*i + 1] = vec_closest_hit_point[i].y;
    closest_hit_point[3*i + 2] = vec_closest_hit_point[i].z;
  }

  cudaFree(d_query_points);
  cudaFree(d_query_distance);
  cudaFree(d_query_hit_points);
  cudaFree(d_query_triID);

}


void distanceField(int num_vertices, double* mesh_vertices,
                   int num_points, double* input_points,
                   double* points_signed_distance,
                   double* closest_hit_point) {

  ////////////////////////////////////////////////////////////
  // lazy way to get "only" distanceField
  signedDistanceField(num_vertices, mesh_vertices, num_points, input_points, points_signed_distance, closest_hit_point);

  for(int i=0; i<num_points; i++) {
    points_signed_distance[i] = std::abs(points_signed_distance[i]);
  }

}

void insideOutside(int num_vertices, double* mesh_vertices,
                   int num_points, double* input_points,
                   double* inside_outside,
                   const int nr_sphere_points = 10) {

  vector<float3> vec_mesh_vertices(num_vertices);
  vector<uint3> vec_mesh_indices(num_vertices);
  for (int i = 0; i < num_vertices; i++) {
    vec_mesh_vertices[i] =
        make_float3(mesh_vertices[3 * i + 0], mesh_vertices[3 * i + 1],
                    mesh_vertices[3 * i + 2]);
  }

  for (int i = 0; i < num_vertices/3; i++) {
    vec_mesh_indices[i] = make_uint3(3*i+0, 3*i+1, 3*i+2);
  }


  ////////////////////////////////////////////////////////////
  // compute inside/outside
  vector<float3> vec_input_points(num_points);
  vector<uint> vec_points_inside_outside(num_points);
  for(int i=0; i<num_points; i++) {
    vec_input_points[i] = make_float3(input_points[3*i+0],
                                      input_points[3*i+1],
                                      input_points[3*i+2]);

  }

  float3* d_input_points;
  cudaMalloc((void**)&d_input_points, sizeof(float3)*vec_input_points.size());
  cudaMemcpy(d_input_points, vec_input_points.data(), sizeof(float3)*vec_input_points.size(), cudaMemcpyHostToDevice);

  int* d_signed_field_results;
  cudaMalloc((void**)&d_signed_field_results, sizeof(int)*vec_input_points.size());
  
  computeInsideOutsidePoints(vec_mesh_vertices, vec_mesh_indices, vec_input_points.size(), d_input_points, d_signed_field_results, nr_sphere_points);

  CUDA_CHECK(cudaMemcpy(vec_points_inside_outside.data(), d_signed_field_results, sizeof(int)*vec_points_inside_outside.size(), cudaMemcpyDeviceToHost));
  ////////////////////////////////////////////////////////////
  // prepare output
  for(int i=0; i<num_points; i++) {
    inside_outside[i] = vec_points_inside_outside[i];
  }

  CUDA_CHECK(cudaFree(d_input_points));
  CUDA_CHECK(cudaFree(d_signed_field_results));


}
