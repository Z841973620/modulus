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

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
project(GeQueL VERSION 0.0.1 LANGUAGES CXX CUDA)

OPTION(GEQUEL_BUILD_TESTS "Build the gequel tests?" OFF)

cmake_minimum_required(VERSION 2.8)

# Git (for auto submodule cloning)
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  # Update submodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
      message(STATUS "Submodule update")
      execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      RESULT_VARIABLE GIT_SUBMOD_RESULT)
      if(NOT GIT_SUBMOD_RESULT EQUAL "0")
          message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
      endif()
  endif()
endif()

find_package(CUDA REQUIRED)

set(owl_dir ${PROJECT_SOURCE_DIR}/submodules/owl)

add_subdirectory(${owl_dir} EXCLUDE_FROM_ALL)

add_definitions(${OWL_CXX_FLAGS})
  
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${owl_dir}/owl/cmake)
include(configure_build_type)

add_subdirectory(gequel)

if(GEQUEL_BUILD_TESTS)
  add_subdirectory(tests)
endif()




