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

# fcp = "find closest point"
cuda_add_executable(gqlTestFCP
  fcp.cu
  OPTIONS ${GEQUEL_OPTIONS}
  )
target_link_libraries(gqlTestFCP
  gequel
  )

# knn = k nearest neighbors
cuda_add_executable(gqlTestKNN
  knn.cu
  OPTIONS ${GEQUEL_OPTIONS}
  )
target_link_libraries(gqlTestKNN
  gequel
  )

# cpat = closest point on any triangle
cuda_add_executable(gqlTestCPAT
  cpat.cu
  OPTIONS ${GEQUEL_OPTIONS}
  )
target_link_libraries(gqlTestCPAT
  gequel
  )


