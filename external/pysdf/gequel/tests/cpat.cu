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

#include "gequel/cpat/cpat.h"
#include "gequel/cpat/cpu/BruteForce.h"
#include "gequel/cpat/cpu/BVH2.h"
#include "gequel/cpat/gpu/BVH2.h"
#include "gequel/common/data_loader.h"

#include <thrust/device_vector.h>
#include <sys/stat.h>

enum QueryPointGenerator {
  UNIFORM,
  CLUSTERED,
  TRIANGLE
};

namespace gequel {  
  extern "C" int main(int ac, char **av)
  {
    // NOTE: These arguments still need incorporating into the below code
    int numQueryPoints = 100000;
    
    QueryPointGenerator generatorType = CLUSTERED;
    bool doSanityTest = true;
    bool useObj = false;
    std::string objPath = "";
    std::string csvEngine = "";
    for (int ai=1;ai<ac;ai++) {
      const std::string arg = av[ai];
      if (arg == "-nst" || arg == "--no-sanity-test")
        doSanityTest = false;
      else if (arg == "-ur")
        generatorType = UNIFORM;
      else if (arg == "-obj") {
        useObj = true;
        objPath = av[++ai];
        struct stat st;
        if (stat(objPath.c_str(), &st) != 0)
          throw std::runtime_error(std::string(objPath + " does not exist!"));
      } else if (arg == "-csv") {
        csvEngine = av[++ai];
        // just write 100 query points to avoid clutter
        numQueryPoints = 100;
      } else if (arg == "-tri") {
        // TODO: specify triangle points somehow (currently hardcoded)
        generatorType = TRIANGLE;
      } else
        throw std::runtime_error("unknown cmdline arg '"+arg+"'");
    }
    if (!useObj) {
      throw std::runtime_error("missing -obj arg");
    }

    std::vector<std::unique_ptr<CPATEngine>> engines;
    engines.emplace_back(new cpat::cpu::BruteForce);
    engines.emplace_back(new cpat::cpu::BVH2EngineBT);
    engines.emplace_back(new cpat::cpu::BVH2EnginePQ);
    engines.emplace_back(new cpat::gpu::BVH2EngineBT);

    vec3f p1(-0.102961, 0.184068, 0.066915);
    vec3f p2(0.086208, 0.032491, 0.022750);
    vec3f p3(-0.052897, 0.058092, -0.067784);
    std::vector<vec3f> queryPoints = createPoints_triangle(p1, p2, p3, numQueryPoints);
    switch (generatorType) {
    case UNIFORM:
      queryPoints = createPoints_uniform(numQueryPoints);
      break;
    case CLUSTERED:
      queryPoints = createPoints_clusters(numQueryPoints);
      break;
    case TRIANGLE:
      queryPoints = createPoints_triangle(p1, p2, p3, numQueryPoints);
      break;
    }

    thrust::device_vector<vec3f> d_queryPoints(queryPoints.begin(), queryPoints.end());

    auto dataSet = createPoints_obj(objPath);
    int numTris = dataSet.second.size();
    std::vector<CPATResult> queryResults(queryPoints.size());
    thrust::device_vector<CPATResult> d_queryResults(queryPoints.size());
    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << "num triangles = " << numTris << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    for (const auto &cpat : engines) {
      // std::cout << "testing engine " << fcp->toString()
      //           << " on #data = " << prettyNumber(dataSet.size()) <<
      //           std::endl;
      try {
        // std::cout << ".... *preparing* data set" << std::endl;
        std::unique_ptr<abstract::PointData> model(cpat->createModel(dataSet.first, dataSet.second));
        bool useDeviceVecs = cpat->getType() == GEQUEL_GPU_NATIVE;

        // warmup
        // std::cout << ".... running queries - warumup" << std::endl;
        if (useDeviceVecs) {
          cpat->findCPATs(thrust::raw_pointer_cast(&d_queryResults[0]),
                         queryPoints.size(), model.get(),
                         thrust::raw_pointer_cast(&d_queryPoints[0]));
        } else {
          cpat->findCPATs(queryResults, model.get(), queryPoints);
        }

        if (useDeviceVecs) {
          thrust::copy(d_queryResults.begin(), d_queryResults.end(),
                       queryResults.begin());
        }
        if (doSanityTest) {
          double sum = 0;
          for (auto result : queryResults)
            sum += result.dist;
          sum /= queryResults.size();
          std::cout << "SANITY CHECK FOR cpat w/ " << cpat->toString() << " : "
                    << sum << std::endl;
        }

        if (cpat->toString() == csvEngine) {
          std::ofstream closestPointsCSV("cpat-closest-points.csv");
          std::ofstream queryPointsCSV("cpat-query-points.csv");
          std::cout << "closest point file: cpat-closest-points.csv\n";
          std::cout << "query point file: cpat-query-points.csv\n";
          for (int i = 0; i < numQueryPoints; i++) {
            queryPointsCSV << queryPoints[i].x << "," << queryPoints[i].y
                           << "," << queryPoints[i].z << std::endl;
            closestPointsCSV << queryResults[i].triID << ","
                             << queryResults[i].point.x << ","
                             << queryResults[i].point.y << ","
                             << queryResults[i].point.z << std::endl;
          }
        }

        // measure
        // std::cout << ".... running queries - measure" << std::endl;
#define AVG_OVER_NUM_SECONDS 3

#ifdef AVG_OVER_NUM_SECONDS
        std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();
        double avgUsPerRun = 0.f;
        int numRuns = 0;
        while (true) {
          if (useDeviceVecs) {
            cpat->findCPATs(thrust::raw_pointer_cast(&d_queryResults[0]),
                            queryPoints.size(), model.get(),
                            thrust::raw_pointer_cast(&d_queryPoints[0]));
          } else {
            cpat->findCPATs(queryResults, model.get(), queryPoints);
          }
          ++numRuns;
          std::chrono::steady_clock::time_point end =
              std::chrono::steady_clock::now();
          double numSecondsSoFar =
              (1.f / 1000000.f) *
              std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
                  .count();
          if (numSecondsSoFar >= double(AVG_OVER_NUM_SECONDS)) {
            avgUsPerRun = 1000 * numSecondsSoFar / numRuns;
            break;
          }
        }

        std::cout << "average time for " << cpat->toString() << " : "
                  << avgUsPerRun << "ms" << std::endl;
#else
        std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();
        if (useDeviceVecs) {
          cpat->findCPATs(thrust::raw_pointer_cast(&d_queryResults[0]),
                          queryPoints.size(), model.get(),
                          thrust::raw_pointer_cast(&d_queryPoints[0]));
        } else {
          cpat->findCPATs(queryResults, model.get(), queryPoints);
        }
        std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();

        std::cout << "time to run " << cpat->toString() << " : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - begin)
                         .count()
                  << "ms" << std::endl;
#endif
      } catch (std::runtime_error e) {
        std::cout << "<<could not test " << cpat->toString()
                  << ">> reason: " << e.what() << std::endl;
      }
    }

    // Test pointTriDistance
    {
      vec3f triA(0.f, 0.f, 0.f);
      vec3f triB(0.f, 1.f, 0.f);
      vec3f triC(1.f, 0.f, 0.f);
      vec3f p1(0.2f, 0.2f, 4.f); // above triangle
      vec3f p2(0.2f, 0.2f, 0.f); // inside triangle
      vec3f p3(1.f, 1.f, 3.f);   // outside triangle - closest to (0.5, 0.5, 0)

      vec3f closest;
      std::cout << "dist to p1: "
                << pointTriDistance(p1, triA, triB, triC, closest) << "@"
                << closest << std::endl;
      std::cout << "dist to p2: "
                << pointTriDistance(p2, triA, triB, triC, closest) << "@"
                << closest << std::endl;
      std::cout << "dist to p3: "
                << pointTriDistance(p3, triA, triB, triC, closest) << "@"
                << closest << std::endl;
    }

    return 0;
  }
  
} //::gequel
