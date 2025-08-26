// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#define TINYOBJLOADER_IMPLEMENTATION

#include "gequel/knn/knn.h"
#include "gequel/knn/cpu/BruteForce.h"
#include "gequel/knn/cpu/KDTree.h"
#include "gequel/knn/gpu/KDTree.h"
#include "gequel/knn/cpu/BVH2.h"
#include "gequel/knn/gpu/BVH2.h"
#include "gequel/common/data_loader.h"

#include <thrust/device_vector.h>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

namespace gequel {
  extern "C" int main(int ac, char **av)
  {
    int numQueryPoints = 100000;
    
    bool uniformRandom = false;
    bool useObj = false;
    bool doSanityTest = true;
    std::string objPath = "";
    std::string csvEngine = "";
    for (int ai=1;ai<ac;ai++) {
      const std::string arg = av[ai];
      if (arg == "-nst" || arg == "--no-sanity-test")
        doSanityTest = false;
      else if (arg == "-ur")
        uniformRandom = true;
      else if (arg == "-obj"){
        useObj = true;
        objPath = av[++ai];
        struct stat st;
        if (stat(objPath.c_str(), &st) != 0)
          throw std::runtime_error(std::string(objPath + " does not exist!"));
      } else if (arg == "-csv") {
        csvEngine = av[++ai];
        // just write 100 query points to avoid clutter
        numQueryPoints = 100;
      } else
        throw std::runtime_error("unknown cmdline arg '"+arg+"'");
    }
    
    std::vector<std::unique_ptr<KNNEngine>> engines;
#if 0
    engines.emplace_back(new knn::cpu::BVH2EngineBT);
    engines.emplace_back(new knn::gpu::BVH2EngineBT);
#else
    engines.emplace_back(new knn::cpu::BruteForce);
    engines.emplace_back(new knn::cpu::KDTree);
    engines.emplace_back(new knn::gpu::KDTree);
    engines.emplace_back(new knn::cpu::BVH2EngineBT);
    engines.emplace_back(new knn::cpu::BVH2EnginePQ);
    engines.emplace_back(new knn::gpu::BVH2EngineBT);
#endif
    std::vector<vec3f> queryPoints
      = uniformRandom
      ? createPoints_uniform(numQueryPoints)
      : createPoints_clusters(numQueryPoints);

    thrust::device_vector<vec3f> d_queryPoints(queryPoints.begin(), queryPoints.end());
    
    std::vector<int> valuesOfKToTest
      /*! more or less arbitrarily picked values, depending on what
          I"ve seen people use in the past */
      = { 10,50,200 };
      // = { 2,5,10,16,20,50,100,200,500 };
    std::vector<float> valuesOfMaxRadiusToTest
      /*! more or less arbitrarily picked values, depending on what
        I"ve seen people use in the past - always relative to length
        of bbox diagonal*/
      = { .02f, 1.f };
      // = { 0.001f, .01f, .05f, .1f, 1.f };
    std::vector<int> numDataPointsToTest;
    if (useObj) {
      numDataPointsToTest.push_back(-1); // -1 meaning, use the obj...
    }
    else {
      numDataPointsToTest = { //10000,
         100000,1000000 };
    }

    std::cout << "query size: " << queryPoints.size() << std::endl;
    
    for (int numDataPoints : numDataPointsToTest) 
    {
      // int numDataPoints = int(.85*(1ull<<logNumPoints));
      // std::vector<vec3f> dataSet = createPoints_uniform(numDataPoints);
      std::vector<vec3f> dataSet;
      if (numDataPoints == -1) {
        auto pointsAndFaces = createPoints_obj(objPath);
        dataSet = std::move(pointsAndFaces.first);
      } else {
        dataSet = uniformRandom
        ? createPoints_uniform(numDataPoints)
        : createPoints_clusters(numDataPoints);//createPoints_clusters(numDataPoints);
      }
      // PRINT(dataSet.size());
      box3f bounds;
      for (auto point : dataSet) bounds.extend(point);
      std::cout << "------------------------------------------------------------------" << std::endl;
      std::cout << "num points = " << dataSet.size() << ", w/ bounds " << bounds << std::endl;
      std::cout << "------------------------------------------------------------------" << std::endl;
      const float modelScale = length(bounds.size());

      for (auto k : valuesOfKToTest) {
        std::vector<KNNResult> queryResults(queryPoints.size() * k);
        thrust::device_vector<KNNResult> d_queryResults(queryPoints.size() * k);
        for (auto maxRadius : valuesOfMaxRadiusToTest)
          for (const auto &knn : engines) {
            try {
              std::unique_ptr<abstract::PointData> model(
                  knn->createModel(dataSet));
              bool useDeviceVecs = knn->getType() == GEQUEL_GPU_NATIVE;

              // warmup
              if (useDeviceVecs) {
                knn->findKNNs(thrust::raw_pointer_cast(&d_queryResults[0]),
                              queryPoints.size(), model.get(),
                              thrust::raw_pointer_cast(&d_queryPoints[0]), k,
                              maxRadius * modelScale);
              } else {
                knn->findKNNs(queryResults, model.get(), queryPoints, k,
                              maxRadius * modelScale);
              }

              if (useDeviceVecs) {
                thrust::copy(d_queryResults.begin(), d_queryResults.end(),
                             queryResults.begin());
              }
              if (doSanityTest) {
                double sum = 0.;

                for (auto result : queryResults)
                  sum += result.dist;
                sum /= queryResults.size();
                std::cout << "SANITY CHECK FOR  k=" << k
                          << " maxR=" << maxRadius * modelScale << " w/ "
                          << knn->toString() << " : " << sum << std::endl;
              }

              // Write csv
              if (knn->toString() == csvEngine) {
                std::stringstream closestPointsCSVName;
                std::stringstream queryPointsCSVName;
                closestPointsCSVName << "knn-closest-points-" << k << "-"
                                     << maxRadius << "-" << numDataPoints
                                     << ".csv";
                queryPointsCSVName << "knn-query-points-" << k << "-" << maxRadius
                                   << "-" << numDataPoints << ".csv";
                std::cout << "closest point file: " << closestPointsCSVName.str() << std::endl;
                std::cout << "query point file: " << queryPointsCSVName.str() << std::endl;
                std::ofstream closestPointsCSV(closestPointsCSVName.str());
                std::ofstream queryPointsCSV(queryPointsCSVName.str());
                for (int i = 0; i < numQueryPoints; i++) {
                  queryPointsCSV << queryPoints[i].x << "," << queryPoints[i].y
                                 << "," << queryPoints[i].z << std::endl;
                  for (int j = 0; j < k; j++) {
                    float inf = std::numeric_limits<float>::infinity();
                    const vec3f point = queryResults[k * i + j].pointID >= 0 ? dataSet[queryResults[k * i + j].pointID] : vec3f(inf, inf, inf);
                    closestPointsCSV << point.x << "," << point.y << "," << point.z << std::endl;
                  }
                }
              }

              // measure

#define AVG_OVER_NUM_SECONDS 3

#ifdef AVG_OVER_NUM_SECONDS
              std::chrono::steady_clock::time_point begin =
                  std::chrono::steady_clock::now();
              double avgMsPerRun = 0.f;
              int numRuns = 0;
              while (true) {
                if (useDeviceVecs) {
                  knn->findKNNs(thrust::raw_pointer_cast(&d_queryResults[0]),
                                queryPoints.size(), model.get(),
                                thrust::raw_pointer_cast(&d_queryPoints[0]), k,
                                maxRadius * modelScale);
                } else {
                  knn->findKNNs(queryResults, model.get(), queryPoints, k,
                                maxRadius * modelScale);
                }
                ++numRuns;
                std::chrono::steady_clock::time_point end =
                    std::chrono::steady_clock::now();
                double numSecondsSoFar =
                    (1.f / 1000000.f) *
                    std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                          begin)
                        .count();
                if (numSecondsSoFar >= double(AVG_OVER_NUM_SECONDS)) {
                  avgMsPerRun = 1000 * numSecondsSoFar / numRuns;
                  break;
                }
              }

              std::cout << "avg for k=" << k
                        << "\tmaxR=" << maxRadius * modelScale << "\tw/ "
                        << knn->toString() << "\t: " << avgMsPerRun << "ms"
                        << std::endl;
#else
              std::chrono::steady_clock::time_point begin =
                  std::chrono::steady_clock::now();
              if (useDeviceVecs) {
                knn->findKNNs(thrust::raw_pointer_cast(&d_queryResults[0]),
                              queryPoints.size(), model.get(),
                              thrust::raw_pointer_cast(&d_queryPoints[0]), k,
                              maxRadius * modelScale);
              } else {
                knn->findKNNs(queryResults, model.get(), queryPoints, k,
                              maxRadius * modelScale);
              }
              std::chrono::steady_clock::time_point end =
                  std::chrono::steady_clock::now();
              std::cout
                  << "time to run k=" << k
                  << "\tmaxR=" << maxRadius * modelScale << "\tw/ "
                  << knn->toString() << "\t: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - begin)
                         .count()
                  << "ms" << std::endl;
#endif
            } catch (std::runtime_error e) {
              std::cout << "<<could not test " << knn->toString()
                        << ">> reason: " << e.what() << std::endl;
            }
          }
      }
    }

    return 0;
  }
  
} //::gequel
