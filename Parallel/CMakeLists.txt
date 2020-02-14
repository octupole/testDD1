cmake_minimum_required( VERSION 3.2) 
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(SOURCE fftw3mpi.cpp)
add_library(parallel STATIC ${SOURCE})
