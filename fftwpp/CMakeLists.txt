cmake_minimum_required( VERSION 3.2) 


if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
set(SOURCE fftw++.cc fftw++.h mpifftw++.cc mpifftw++.h mpitranspose.cc mpitranspose.h)
add_library(fftwppmpi STATIC ${SOURCE})
