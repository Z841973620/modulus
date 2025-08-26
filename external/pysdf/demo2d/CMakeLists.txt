project(sdf_demo2d LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 3.17)

find_package(CUDAToolkit REQUIRED)

add_executable(demo2d
  main.cpp
  )

target_include_directories(demo
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/
  )

target_link_libraries(demo
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/libsdf.so
  CUDA::cudart_static
  CUDA::nvToolsExt
  )
