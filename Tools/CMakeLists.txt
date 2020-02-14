cmake_minimum_required( VERSION 3.2) 


if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffast-math -fno-math-errno") 
set(SOURCE MyUtilClass.cpp MyTools.cpp)
add_library(tools STATIC ${SOURCE})
