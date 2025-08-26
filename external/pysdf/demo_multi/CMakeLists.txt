project(sdf_demo LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 3.17)

find_package(CUDAToolkit REQUIRED)

add_executable(demo_multi
  main.cpp
  )

target_include_directories(demo
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/
  )

target_link_libraries(demo
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/libsdf.so
  CUDA::cudart_static
  )
