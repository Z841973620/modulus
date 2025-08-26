# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

cmake_minimum_required(VERSION 3.17)

project(optix_sdf LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDAToolkit REQUIRED)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

if(WIN32)
	add_compile_definitions(NOMINMAX)
else()
  set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
  set(CMAKE_SHARED_LINKER_FLAGS "-static-libgcc -static-libstdc++")
endif()

if(WIN32)
  set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0/" CACHE PATH "OPTIX7 SDK location for libsdf and sublibraries")
else()
  set(OptiX_INSTALL_DIR "$ENV{HOME}/optix7" CACHE PATH "OPTIX7 SDK location for libsdf and sublibraries")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(gequel)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/gequel/submodules/owl/owl/cmake)

# provides ${OptiX_INCLUDE}
find_package(OptiX REQUIRED VERSION 7)

# build ".ptx" files (a cmake "object" library)
add_library(sdf_ptx OBJECT src/grid.cu)


target_include_directories(sdf_ptx
  PRIVATE
  ${OptiX_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
set_property(TARGET sdf_ptx PROPERTY CUDA_PTX_COMPILATION ON)

add_library(pysdf SHARED
  src/sdf.cpp
  src/cpat.cpp
  src/cpat.cu
  src/knn.cpp
  src/prq.cpp
  )

target_compile_options(pysdf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(pysdf PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>) 

add_executable(test_cpat
  src/test_cpat.cpp
  )

add_executable(demo
  demo/main.cpp
  )

add_executable(demo_multi
  demo_multi/main.cpp
  )


add_executable(demo2d
  demo2d/main.cpp
  demo2d/cli_reader.cpp
  )

set_property(TARGET pysdf PROPERTY POSITION_INDEPENDENT_CODE ON)

add_dependencies(pysdf sdf_ptx)

# create resource header from PTX using `bin2c`, which creates char array 'grid_ptx_storage' of the PTX file stored in file 'grid.h'
add_custom_target(sdf_ptx_header
  COMMAND bin2c --name grid_ptx_storage --const $<TARGET_OBJECTS:sdf_ptx> -p 0 > ${PROJECT_BINARY_DIR}/grid.h 
  )

add_dependencies(sdf_ptx_header sdf_ptx)

target_link_libraries(pysdf
  CUDA::cudart_static
  CUDA::nvToolsExt
  gequel
  )

target_include_directories(pysdf
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OptiX_INCLUDE}
  ${PROJECT_BINARY_DIR}
  )


add_dependencies(pysdf sdf_ptx_header)

target_link_libraries(test_cpat
  pysdf
  )

target_include_directories(test_cpat
  PRIVATE
  )

target_link_libraries(demo
  pysdf
  )

target_link_libraries(demo2d
  pysdf
  )

target_link_libraries(demo_multi
  pysdf
  )

target_include_directories(demo2d
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/demo2d
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  )

install(TARGETS demo
  EXPORT demo
  RUNTIME DESTINATION bin)

install(TARGETS pysdf
  LIBRARY DESTINATION lib
  )

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/sdf.h
  DESTINATION include
  )

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/bunny.obj
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/tiny_obj_loader.h
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/CMakeLists.txt
  DESTINATION demo_src
  )

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/demo/README.md
  DESTINATION .
  )

set(CPACK_GENERATOR "ZIP" CACHE STRING "Generators to support. semi-colon delimited list")
set(CPACK_PACKAGE_NAME "libpysdf")
set(CPACK_PACKAGE_VERSION 0.1)

include(CPack)
