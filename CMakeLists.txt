cmake_minimum_required( VERSION 3.0) 
# Project name
project(testDD CXX C)

set(CMAKE_CXX_STANDARD 14)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_SOURCE_DIR}/GreatCMakeCookOff/modules)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules"  "${CMAKE_SOURCE_DIR}/GreatCMakeCookOff/modules")

include(macros)


if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

include(CheckCXXCompilerFlag)
include(CheckIncludeFileCXX)
include(CheckCXXSymbolExists)
include(CMakeHostSystemInformationExtra)
LONG_TODAY(BUILD_DATE)


detect_x64_micro_architecture(VENDOR ARCHITECTURE)
string(TIMESTAMP BUILD_DATE "%b %d %Y %H:%M:%S" )

option(PARALLEL "Enable Parallel compilation " ON)

find_package(MPI)
if (NOT MPI_CXX_FOUND)
	message(STATUS "========== testDD Parallel Information ==========")
 	message(STATUS )
  	message(STATUS "No MPI compiler found. Perhaps you want to set one explicitly?")
  	message(STATUS "To override the default compiler, set the environment variable")
  	message(STATUS "     export CXX=/full/path/to/mpic++")
  	message(STATUS "in bash or ksh or sh.  In csh or tcsh, use:")
  	message(STATUS "     setenv CXX /full/path/to/mpic++")
  	message(STATUS "before running the cmake command.")
  	message(STATUS )
  	message(STATUS "==================================================")
  	message(FATAL_ERROR "Cannot compile without MPI!")
else( MPI_CXX_FOUND )
  	INCLUDE_DIRECTORIES(${MPI_CXX_INCLUDE_PATH})
  	if(DEFINED $ENV{CXX})
    	message(STATUS "Found MPI compiler, but using the CXX compiler set in the CXX environment variable, $ENV{CXX}")
  	else()
    	set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    	set(ENV{CXX} ${MPI_CXX_COMPILER})
  	endif()
	add_definitions(-DHAVE_MPI=1)
endif()
include(FindOpenMP)
if(MPI_CXX_FOUND AND OpenMP_CXX_FLAGS)
	find_package(FFTW3 COMPONENTS DOUBLE OPENMP)
	set(FFTW3_MYLIBRARIES ${FFTW3_DOUBLE_OPENMP_LIBRARY}  ${FFTW3_DOUBLE_SERIAL_LIBRARY})
elseif(MPI_CXX_FOUND)
	find_package(FFTW3 COMPONENTS DOUBLE )
	set(FFTW3_MYLIBRARIES ${FFTW3_DOUBLE_SERIAL_LIBRARY})
endif()

if (OpenMP_CXX_FLAGS)
	add_definitions(-DHAVE_OPENMP=1)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.2)
        message(FATAL_ERROR "Clang version must be at least 3.2!")
    endif()
    set(CMAKE_CXX_FLAGS "-std=c++14 -O3 ${OpenMP_CXX_FLAGS}")        ## Optimize	
    message(WARNING "Not tested, use at your won risk.")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
   set(CMAKE_CXX_FLAGS "-std=c++14 -ffast-math -fno-math-errno ${OpenMP_CXX_FLAGS}")  
   message(FATAL_ERROR "Cannot compile, AppleClang up to LLVM version 9.1.0 gives an internal error!")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
       message(FATAL_ERROR "Intel icc version must be >= 17. It is ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
   set(CMAKE_CXX_FLAGS "-std=c++14 ${OpenMP_CXX_FLAGS}")  

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.4)
        message(FATAL_ERROR "GCC version must be at least 5.4")
    endif()
   set(CMAKE_CXX_FLAGS "-std=c++14 -O3 -ffast-math -fno-math-errno ${OpenMP_CXX_FLAGS}")  
endif()

include_directories("${PROJECT_SOURCE_DIR}/fftwpp")
include_directories("${FFTW3_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}/abseil-cpp")
add_subdirectory(abseil-cpp)
add_subdirectory(fftwpp)
add_subdirectory(Tools)

set(Abseil_Libraries absl::strings absl::hash absl::flags absl::flags_parse)
add_executable(testDD.x main.cpp )

target_link_libraries(testDD.x tools fftwppmpi ${FFTW3_MYLIBRARIES} ${Abseil_Libraries})

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "$ENV{HOME}" CACHE PATH "default install path" FORCE )
endif()
message( STATUS "LIBRARIES ........... = ${PFFT_LIBRARY}")
message( STATUS)
message( STATUS "========== OpenTRJ Build Information ===========")
message( STATUS "Current date/time .......... = ${BUILD_DATE}")
message( STATUS "CMAKE_SYSTEM ............... = ${CMAKE_SYSTEM}")
message( STATUS "================================================")
message( STATUS "CMAKE_BUILD_TYPE ........... = ${CMAKE_BUILD_TYPE}")
message( STATUS "CMAKE_INSTALL_PREFIX ....... = ${CMAKE_INSTALL_PREFIX}")
message( STATUS "CMAKE_C_COMPILER ........... = ${CMAKE_C_COMPILER}")
message( STATUS "CMAKE_CXX_COMPILER ......... = ${CMAKE_CXX_COMPILER}")
if(MPI_CXX_FOUND)
message( STATUS "MPI_CXX_COMPILER ........... = ${MPI_CXX_COMPILER}")
message( STATUS "MPI_CXX_INCLUDE_PATH ....... = ${MPI_CXX_INCLUDE_PATH}")
message( STATUS "MPI_CXX_LIBRARIES .......... = ${MPI_CXX_LIBRARIES}")
message( STATUS "MPI_CXX_VERSION ............ = ${MPI_CXX_VERSION}")
message( STATUS "MPI_COMPILE_FLAGS .......... = ${CMAKE_CXX_FLAGS}")
endif()
if (OpenMP_CXX_FLAGS)
message(STATUS  "Compiled with OPENMP ....... = YES")
else()
message(STATUS  "Compiled with OPENMP ....... = NO ")
endif()
if(FFTW3_FOUND)
message( STATUS "FFTW3_INCLUDE_DIR .......... = ${FFTW3_INCLUDE_DIRS}")
message( STATUS "FFTW3_LIBRARIES ............ = ${FFTW3_LIBRARIES}")
if(FFTW3_DOUBLE_OPENMP_FOUND) 
message( STATUS "FFTW3_DOUBLE_OPENMP ......... = ${FFTW3_DOUBLE_OPENMP_LIBRARY}")
endif()
if(FFTW3_DOUBLE_THREADS_FOUND) 
message( STATUS "FFTW3_DOUBLE_THREADS ........ = ${FFTW3_DOUBLE_THREADS_LIBRARY}")
endif()
else()
message( STATUS  "FFTW3            .......... = NOT FOUND")
endif()
message( STATUS)
message( STATUS "To override these options, add -D{OPTION_NAME}=... to the cmake command" )
message( STATUS "Particularly useful defines are for:")
message( STATUS )
message( STATUS "  -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}	(where OpenTRJ executables will be installed)")
message( STATUS "  -DHINT_DIRS=/path/to/fftw3	(select path to installed FFTW3 routines)")
message( STATUS "  -DPARALLEL=OFF	(Do not compile with the default PARALLEL=ON)")
message( STATUS )
