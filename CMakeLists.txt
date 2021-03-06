# AUTOMOC requires 2.8.6. Per-target include directories would require 2.8.8
cmake_minimum_required(VERSION 2.8.8)
project(gcal)

message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  (This can be changed with `cmake -DCMAKE_INSTALL_PREFIX=/some/place`")

# From CMAKE_SYSTEM work out which of __OSX__, __GLN__, __NIX__ or
# __WIN__ are required
message(STATUS "Operating system: " ${CMAKE_SYSTEM})
if(CMAKE_SYSTEM MATCHES Linux.*)
set(MORPH_HOST_DEFINITION "-D__GLN__")
elseif(CMAKE_SYSTEM MATCHES BSD.*)
set(MORPH_HOST_DEFINITION "-D__NIX__")
elseif(APPLE)
set(MORPH_HOST_DEFINITION "-D__OSX__")
elseif(CMAKE_SYSTEM MATCHES Win.*)
set(MORPH_HOST_DEFINITION "-D__WIN__")
else()
message(ERROR "Operating system not supported: " ${CMAKE_SYSTEM})
endif()

# Add the host definition to CXXFLAGS along with other switches
if (APPLE)
# OpenMP: According to
# https://stackoverflow.com/questions/39979836/using-openmp-with-c11-on-mac-os#39983933
# it should now be a case of:
#
# You should be able to get away without the libomp that's mentioned
# in that post, as I don't use any of the runtime library OpenMP
# features. However, the code did not compile without it, so we've moved the links to omp to the later apple if case.
#
set(CMAKE_CXX_FLAGS "${MORPH_HOST_DEFINITION} -Wall -g -std=c++11 -Xpreprocessor -fopenmp")

else()
# To use Intel compiler, call cmake as: cmake -DCMAKE_CXX_COMPILER=icpc ..
if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
set(CMAKE_CXX_FLAGS "${MORPH_HOST_DEFINITION} -Wall -g -std=c++11 -xHOST -O3 -qopenmp -D__ICC__")
# No OMP; it goes slower:
#set(CMAKE_CXX_FLAGS "${MORPH_HOST_DEFINITION} -Wall -g -std=c++11 -xHOST -O3 -D__ICC__")
# No optimisation at all:
#set(CMAKE_CXX_FLAGS "${MORPH_HOST_DEFINITION} -Wall -g -std=c++11 -xHOST -O0  -D__ICC__")
else() # GCC
set(CMAKE_CXX_FLAGS "${MORPH_HOST_DEFINITION} -Wall -g -std=c++11 -Wno-unused-result -Wno-unknown-pragmas -march=native -O3 -fopenmp")
endif()
endif()

set(CMAKE_CXX_STANDARD 11)
find_package(absl REQUIRED)

# Lib finding - we have to do this, so that the correct include paths
# are present for OpenGL, OpenCV, etc; they're all included by
# morphologica header files.
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(X11 REQUIRED)

if(APPLE)
find_package (HDF5 REQUIRED)
set (HDF5LIBLINK ${HDF5_C_LIBRARY_hdf5})
set (JSONLIBLINK /usr/local/lib/libjsoncpp.dylib)
include_directories(/opt/local/include/libomp)
link_libraries(-L/opt/local/lib/libomp omp)
else()
set (LIB_TYPE SHARED) # STATIC or SHARED
string(TOLOWER ${LIB_TYPE} SEARCH_TYPE)
endif(APPLE)

include_directories(${CMAKE_INSTALL_PREFIX}/include)

# Find libmorphologica using pkg-config
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
message(STATUS "Have pkg-config, searching for libmorphologica...")

set(ENV{PKG_CONFIG_PATH}  "ENV{PKG_CONFIG_PATH}:${CMAKE_INSTALL_PREFIX}/lib/pkgconfig")

pkg_check_modules(MORPH REQUIRED libmorphologica)
if (MORPH_FOUND)
message(STATUS "pkg-config found libmorphologica")
include_directories(${MORPH_INCLUDEDIR})
else()
message(FATAL_ERROR "You need libmorphologica from github.com/ABRG_Models/morphologica")
endif(MORPH_FOUND)
else()
message(WARNING "There is no pkg-config on this system to check for libmorphologica.")
find_package (HDF5 NAMES hdf5 COMPONENTS C ${SEARCH_TYPE} REQUIRED)
set (HDF5LIBLINK ${HDF5_C_${LIB_TYPE}_LIBRARY})
set (JSONLIBLINK jsoncpp)
endif()

add_subdirectory(sim)

# For debugging of variables:
set(DEBUG_VARIABLES OFF)
if(DEBUG_VARIABLES)
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
endif(DEBUG_VARIABLES)

