# ======================================================================== #
# Copyright 2020 Ingo Wald                                                 #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

if(OWL_USE_TBB)
set(GEQUEL_OPTIONS
  -DOWL_HAVE_TBB=1
)
endif()


set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_75,code=compute_75 --use_fast_math -lineinfo")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(gequel
  fcp/cpu/BruteForce.cpp
  fcp/cpu/KDTree.cpp
  fcp/gpu/KDTree.cu
  fcp/gpu/BVH2.cu

  knn/cpu/BruteForce.cpp
  knn/cpu/KDTree.cpp
  knn/gpu/KDTree.cu
  knn/cpu/BVH2.cpp
  knn/gpu/BVH2.cu

  cpat/cpu/BruteForce.cpp
  cpat/cpu/BVH2.cpp
  cpat/gpu/BVH2.cu

  common/cpu/datastructures/bvh2.cpp
  )
target_link_libraries(gequel
  ${OWL_LIBRARIES}
  )

target_include_directories(gequel
  PUBLIC
  ${OWL_INCLUDES}
  ${PROJECT_SOURCE_DIR}
  )
