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

#include "gequel/fcp/fcp.h"
#include "gequel/fcp/cpu/BruteForce.h"
#include "gequel/fcp/cpu/KDTree.h"
#include "gequel/fcp/gpu/KDTree.h"
#include "gequel/fcp/gpu/BVH2.h"
#include "gequel/common/data_loader.h"

#include <thrust/device_vector.h>
#include <sys/stat.h>

namespace gequel {  
  extern "C" int main(int ac, char **av)
  {
    // NOTE: These arguments still need incorporating into the below code
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
        numQueryPoints = 100;
      } else
        throw std::runtime_error("unknown cmdline arg '"+arg+"'");
    }

    std::vector<std::unique_ptr<FCPEngine>> engines;
    engines.emplace_back(new fcp::cpu::BruteForce);
    engines.emplace_back(new fcp::cpu::KDTree);
    engines.emplace_back(new fcp::gpu::KDTree);
    engines.emplace_back(new fcp::gpu::BVH2EngineBT);

    std::vector<int> numDataPointsToTest;
    numDataPointsToTest = {1000,    2000,    5000,    10000,  20000,
                           50000,   100000,  200000,  500000, 1000000,
                           2000000, 5000000, 10000000};

    std::vector<vec3f> queryPoints = createPoints_clusters(numQueryPoints);
    if (useObj) {
      box3f bounds = makeAABBForOBJ(objPath);
      rescaleData(queryPoints, bounds);
    }
    thrust::device_vector<vec3f> d_queryPoints(queryPoints.begin(), queryPoints.end());

    std::ofstream timingFile("fcp-times.csv");
    timingFile << "size";
    for (const auto& fcp : engines) {
      timingFile << "," << fcp->toString();
    }
    timingFile << std::endl;

    for (int numDataPoints : numDataPointsToTest)
    {
      std::vector<vec3f> dataSet;
      if (useObj) {
        dataSet = createPoints_triangles(objPath, numDataPoints);
      } else {
        dataSet = uniformRandom
        ? createPoints_uniform(numDataPoints)
        : createPoints_clusters(numDataPoints);
      }
      {
        std::ofstream dataFile("data.csv");
        dataFile << "x,y,z\n";
        for (vec3f v : dataSet) {
          dataFile << v.x << "," << v.y << "," << v.z << std::endl;
        }
      }
      timingFile << dataSet.size();
      std::vector<FCPResult> queryResults(queryPoints.size());
      thrust::device_vector<FCPResult> d_queryResults(queryPoints.size());
      std::cout << "------------------------------------------------------------------" << std::endl;
      std::cout << "num points = " << dataSet.size() << std::endl;
      std::cout << "------------------------------------------------------------------" << std::endl;
      for (const auto& fcp : engines) {
        // std::cout << "testing engine " << fcp->toString()
        //           << " on #data = " << prettyNumber(dataSet.size()) << std::endl;
        try {
          //std::cout << ".... *preparing* data set" << std::endl;
          std::unique_ptr<abstract::PointData> model(fcp->createModel(dataSet));
          bool useDeviceVecs = fcp->getType() == GEQUEL_GPU_NATIVE;
          
          // warmup
          //std::cout << ".... running queries - warumup" << std::endl;
          if (useDeviceVecs) {
            fcp->findClosestPoint(thrust::raw_pointer_cast(&d_queryResults[0]),
                                  queryPoints.size(), model.get(),
                                  thrust::raw_pointer_cast(&d_queryPoints[0]));
          } else {
            fcp->findClosestPoint(queryResults, model.get(), queryPoints);
          }

          if (useDeviceVecs) {
            thrust::copy(d_queryResults.begin(), d_queryResults.end(), queryResults.begin());
          }
          if (doSanityTest) {
            double sum = 0;
            for (auto result : queryResults)
              sum += result.distance;
            sum /= queryResults.size();
            std::cout << "SANITY CHECK FOR fcp w/ " << fcp->toString() << " : " << sum << std::endl;
          }

          if (fcp->toString() == csvEngine) {
            std::stringstream closestPointsCSVName;
            std::stringstream queryPointsCSVName;
            closestPointsCSVName << "fcp-closest-points-" << numDataPoints << ".csv";
            queryPointsCSVName << "fcp-query-points-" << numDataPoints << ".csv";
            std::cout << "closest point file: " << closestPointsCSVName.str() << std::endl;
            std::cout << "query point file: " << queryPointsCSVName.str() << std::endl;
            std::ofstream closestPointsCSV(closestPointsCSVName.str());
            std::ofstream queryPointsCSV(queryPointsCSVName.str());
            for (int i = 0; i < numQueryPoints; i++) {
              queryPointsCSV << queryPoints[i].x << "," << queryPoints[i].y
                             << "," << queryPoints[i].z << std::endl;
              const vec3f point = dataSet[queryResults[i].pointID];
              closestPointsCSV << point.x << "," << point.y << "," << point.z << std::endl;
            }
          }

          // measure
          //std::cout << ".... running queries - measure" << std::endl;
#define AVG_OVER_NUM_SECONDS 1

#ifdef AVG_OVER_NUM_SECONDS
          std::chrono::steady_clock::time_point begin =
              std::chrono::steady_clock::now();
          double avgUsPerRun = 0.f;
          int numRuns = 0;
          while (true) {
            if (useDeviceVecs) {
              fcp->findClosestPoint(
                  thrust::raw_pointer_cast(&d_queryResults[0]),
                  queryPoints.size(), model.get(),
                  thrust::raw_pointer_cast(&d_queryPoints[0]));
            } else {
              fcp->findClosestPoint(queryResults, model.get(), queryPoints);
            }
            ++numRuns;
            std::chrono::steady_clock::time_point end =
                std::chrono::steady_clock::now();
            double numSecondsSoFar =
              (1.f/1000000.f)*
                std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                    .count();
            if (numSecondsSoFar >= double(AVG_OVER_NUM_SECONDS)) {
              avgUsPerRun = 1000 * numSecondsSoFar / numRuns;
              break;
            }
          }

          std::cout << "average time for " << fcp->toString() << " : "
                    << avgUsPerRun << "ms" << std::endl;

          timingFile << "," << avgUsPerRun;
#else
          std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
          if (useDeviceVecs) {
            fcp->findClosestPoint(thrust::raw_pointer_cast(&d_queryResults[0]),
                                  queryPoints.size(), model.get(),
                                  thrust::raw_pointer_cast(&d_queryPoints[0]));
          } else {
            fcp->findClosestPoint(queryResults, model.get(), queryPoints);
          }
          std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
          
          std::cout << "time to run " << fcp->toString() << " : "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                    << "ms" << std::endl;
#endif
        } catch (std::runtime_error e) {
          std::cout << "<<could not test " << fcp->toString() << ">> reason: " << e.what() << std::endl;
        }
      }
      timingFile << std::endl;
    }
    
    return 0;
  }
  
} //::gequel
