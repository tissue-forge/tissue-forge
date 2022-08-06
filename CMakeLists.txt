#*******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022 T.J. Sego
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
#*******************************************************************************

# need 3.20.3 for CUDA fixes
cmake_minimum_required (VERSION 3.20.3)


if(APPLE)

  message("CMAKE_OSX_DEPLOYMENT_TARGET: ${CMAKE_OSX_DEPLOYMENT_TARGET}")

  if("${CMAKE_OSX_DEPLOYMENT_TARGET}" VERSION_GREATER_EQUAL "10.15")

    message("MacOS newer than 10.15, building universal binary")

    set (CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
    set (CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO)

  else()

    message("MacOS older than 10.15, not building universal binary")

  endif()

endif()

project(TissueForge)

# sets the target types
if(UNIX)
  # Both APPLE and UNIX are defined on OSX
  if(APPLE)
    set(TF_APPLE 1)
  else()
    set(TF_LINUX 1)
  endif()
  set(TF_UNIX 1)
elseif(WIN32)
  set(TF_WINDOWS 1)
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# handle rpath
set(CMAKE_MACOSX_RPATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
if(APPLE)
  set(RPATH_PREFIX @loader_path)
else()
  set(RPATH_PREFIX $ORIGIN)
endif()
file(RELATIVE_PATH RDIR
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR} 
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
)
set(CMAKE_INSTALL_RPATH ${RPATH_PREFIX} ${RPATH_PREFIX}/${RDIR})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(TF_OPENMP "Enable OpenMP parallelism" OFF)

if(TF_OPENMP)
  find_package(OpenMP)

  message("OpenMP_FOUND: ${OpenMP_FOUND}")

  message("OpenMP_CXX_INCLUDE_DIRS: ${OpenMP_CXX_INCLUDE_DIRS}")

  message("OpenMP_CXX_LIBRARIES: ${OpenMP_CXX_INCLUDE_DIRS}")

  option(TF_OPENMP_BONDS "Enable OpenMP bonds" OFF)
  option(TF_OPENMP_INTEGRATOR "Enable OpenMP integrator" OFF)

endif()

option(TF_VECTORIZE_FLUX       "Enable vector flux calculations (AVX instructions)" OFF)
option(TF_VECTORIZE_FORCE      "Enable vector force calculations (AVX instructions)" OFF)
option(TF_VECTORIZE_BONDS      "Enable vector bonded interactions (AVX instructions)" OFF)
option(TF_VECTORIZE_INTEGRATOR "Enable vector integrator  (AVX instructions)" OFF)
option(TF_ENABLE_AVX2          "Enable AVX2 instructions" OFF)
option(TF_ENABLE_AVX           "Enable AVX  instructions" ON)
option(TF_ENABLE_SSE4          "Enable SSE4 instructions" ON)


if(TF_ENABLE_AVX2)
  set(TF_AVX2 ON)
  set(TF_AVX ON)
  set(TF_SSE42 ON)
  set(TF_SIMD_SIZE 8)
  add_compile_options(-mavx)
  add_compile_options(-mavx2)
  add_compile_options(-msse4.2)
elseif(TF_ENABLE_AVX)
  set(TF_AVX ON)
  set(TF_SSE42 ON)
  set(TF_SIMD_SIZE 8)
  add_compile_options(-mavx)
  add_compile_options(-msse4.2)
elseif(TF_ENABLE_SSE4)
  set(TF_SSE42 ON)
  set(TF_SIMD_SIZE 4)
  add_compile_options(-msse4.2)
else()
  set(TF_SIMD_SIZE 0)
endif()

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  message(STATUS "Detected CUDA")
  message(STATUS "  Version           : ${CUDAToolkit_VERSION}")
  message(STATUS "  Architectures     : ${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "  Compiler          : ${CUDAToolkit_NVCC_EXECUTABLE}")
  set(TF_HAVECUDA ON)
else()
  message(STATUS "No detected CUDA support")
  set(TF_HAVECUDA OFF)
endif()

set(TF_INCLUDE_DIR include/TissueForge)
set(TF_PRIVATE_INCLUDE_DIR ${TF_INCLUDE_DIR}/private)
set(TF_MODELS_INCLUDE_DIR ${TF_INCLUDE_DIR}/models)
set(TF_RESOURCE_DIR resources/TissueForge)
set(TF_SHARE_DIR share/cmake/TissueForge)

if(TF_HAVECUDA)
  enable_language(CUDA)
  set(TF_CUDA_DIR ${TF_RESOURCE_DIR}/cuda)
  set(TF_CUDA_INCLUDE_DIR ${CUDAToolkit_INCLUDE_DIRS})
  option(TF_WITHCUDA             "Enable GPU-accelerated calculations using CUDA" OFF)
  option(TF_CUDA_DEBUG           "Enable CUDA debug mode" OFF)
  if(DEFINED ENV{TF_WITHCUDA})
    set(TF_WITHCUDA ON)
  endif()
endif()

# Find assimp before magnum-plugins to avoid fixes that don't seem relevant for current versions
find_package(assimp REQUIRED)

if(assimp_FOUND)
  message("Assimp found: ${ASSIMP_INCLUDE_DIRS}")
else()
  message("Assimp not found")
endif()

# Borrowing Magnum's modules. 
list(APPEND CMAKE_MODULE_PATH 
  ${PROJECT_SOURCE_DIR}/extern/magnum/modules
  ${PROJECT_SOURCE_DIR}/extern/magnum-plugins/modules
)


file(READ "VERSION.txt" ver)

string(REGEX MATCH "VERSION_MAJOR ([0-9]*)" _ ${ver})
set(TF_VERSION_MAJOR ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_MINOR ([0-9]*)" _ ${ver})
set(TF_VERSION_MINOR ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_PATCH ([0-9]*)" _ ${ver})
set(TF_VERSION_PATCH ${CMAKE_MATCH_1})

string(REGEX MATCH "VERSION_DEV ([0-9]*)" _ ${ver})
set(TF_VERSION_DEV ${CMAKE_MATCH_1})


message("version: ${TF_VERSION_MAJOR}.${TF_VERSION_MINOR}.${TF_VERSION_PATCH}")

# Get the build qualifier from the environment, if one was specified
if(DEFINED ENV{TFBUILDQUAL})
  set(TF_BUILDQUAL $ENV{TFBUILDQUAL})
  set(TF_HASBUILDQUAL true)
  message("qualifer: ${TF_BUILDQUAL}")
else()
  set(TF_HASBUILDQUAL false)
endif()

# Get package details
#   Local install is assumed unless TFPACKAGELOCALOFF env var is set
set(TF_PACKAGE_LOCAL true)
if(DEFINED ENV{TFPACKAGELOCALOFF})
  set(TF_PACKAGE_LOCAL false)
  message("Disabled local package")
else()
  message("Enabled local package")
endif()
#   Conda is enabled when TFPACKAGECONDA env var is set
set(TF_PACKAGE_CONDA false)
if(DEFINED ENV{TFPACKAGECONDA})
  set(TF_PACKAGE_CONDA true)
  message("Enabled conda package")
endif()

message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
message("CMAKE_OSX_DEPLOYMENT_TARGET: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message("MACOSX_DEPLOYMENT_TARGET: ${MACOSX_DEPLOYMENT_TARGET}")
message("CMAKE_OSX_SYSROOT: ${CMAKE_OSX_SYSROOT}")
message("XCODE_VERSION: ${XCODE_VERSION}")
message("CMAKE_GENERATOR: ${CMAKE_GENERATOR}")
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message("CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

set(TF_HAVEPTHREADS FALSE)
if(DEFINED PThreads_ROOT)

  find_library(
    PThreads 
    NAMES pthreads pthreadsVSE2 
    HINTS ${PThreads_ROOT} 
    REQUIRED 
  )
  message("PThreads path: ${PThreads}")
  add_library(pthreads UNKNOWN IMPORTED GLOBAL)
  target_include_directories(pthreads INTERFACE ${PThreads_ROOT}/include)
  set_target_properties(pthreads PROPERTIES IMPORTED_LOCATION ${PThreads})
  add_library(Threads::Threads ALIAS pthreads)
  set(TF_HAVEPTHREADS TRUE)

else()
  # use standard unix pthreads, creates Threads::Threads target
  find_package(Threads REQUIRED)
endif()

# Install options

set(CMAKE_INSTALL_PREFIX  "$ENV{HOME}/local/TissueForge"  CACHE PATH  "Install location for Tissue Forge")
set(BUILD_SHARED_LIBS     ON                              CACHE BOOL  "build shared/dynamic library")
set(CMAKE_INFO_PREFIX     "${CMAKE_INSTALL_PREFIX}"       CACHE PATH  "Install location for Tissue Forge basic info")

# Language options

option(TF_WITH_C        "Generate the C language interface for Tissue Forge."      ON)
option(TF_WITH_PYTHON   "Generate the Python language interface for Tissue Forge." ON)

if(TF_WITH_C) 

  set(TF_WRAPS_C_INSTALL_PREFIX include/TissueForge/wraps/C)

endif()

# Precision options
option(TF_FPTYPE_SINGLE "Build Tissue Forge with single precision."      ON)

# Enforce single precision for CUDA support
if(TF_HAVECUDA)
  if(TF_WITHCUDA)
    if(NOT TF_FPTYPE_SINGLE)
      message(WARNING "CUDA-supporting installations only support single precision.")
      set(TF_FPTYPE_SINGLE ON)
    endif()
  endif()
endif()

# Python setup

if(TF_WITH_PYTHON)

  # directory where we build the Tissue Forge python package
  set(TF_PYPKG_DIR ${CMAKE_BINARY_DIR}/TissueForge)

  find_package(Python 3.7.0...3.9.99 REQUIRED COMPONENTS Interpreter Development)

  message("Python_FOUND: ${Python_FOUND}")
  message("Python_Interpeter_FOUND: ${Python_Interpeter_FOUND}")
  message("Python_VERSION: ${Python_VERSION}")
  message("Python_Development_FOUND: ${Python_Development_FOUND}")

  message("Python_Development_FOUND: ${Python_Development_FOUND}")
  message("Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
  message("Python_LIBRARIES: ${Python_LIBRARIES}")
  message("Python_LIBRARY_DIRS: ${Python_LIBRARY_DIRS}")
  message("Python_RUNTIME_LIBRARY_DIRS: ${Python_RUNTIME_LIBRARY_DIRS}")
  message("Python_EXECUTABLE: ${Python_EXECUTABLE}")

  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var(name='prefix'))" 
    OUTPUT_VARIABLE PY_ROOT_DIR
  )
  string(STRIP ${PY_ROOT_DIR} PY_ROOT_DIR)
  file(TO_CMAKE_PATH "${PY_ROOT_DIR}" PY_ROOT_DIR)
  message("PY_ROOT_DIR: ${PY_ROOT_DIR}")

  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path(name='platlib'))" 
    OUTPUT_VARIABLE PY_SITEPACKAGES
  )
  string(STRIP ${PY_SITEPACKAGES} PY_SITEPACKAGES)
  file(TO_CMAKE_PATH "${PY_SITEPACKAGES}" PY_SITEPACKAGES)
  message("PY_SITEPACKAGES: ${PY_SITEPACKAGES}")

  if(Python_Development_FOUND)
    get_target_property(TMP Python::Python IMPORTED_LOCATION_DEBUG)
    message("Python IMPORTED_LOCATION_DEBUG: ${TMP}")

    get_target_property(TMP Python::Python IMPORTED_LOCATION_RELEASE)
    message("Python IMPORTED_LOCATION_RELEASE: ${TMP}")
  endif()

  # Determine install site-packages location using info from environment if not provided
  if(DEFINED TF_INSTALL_PREFIX_PYTHON)
    SET(TF_SITEPACKAGES ${TF_INSTALL_PREFIX_PYTHON})
  else()
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import sysconfig, os; print(os.path.relpath(sysconfig.get_path(name='platlib'), sysconfig.get_config_var(name='prefix')))" 
      OUTPUT_VARIABLE TF_SITEPACKAGES_REL
    )
    string(STRIP ${TF_SITEPACKAGES_REL} TF_SITEPACKAGES_REL)
    file(TO_CMAKE_PATH "${TF_SITEPACKAGES_REL}" TF_SITEPACKAGES_REL)
    set(TF_SITEPACKAGES ${CMAKE_INSTALL_PREFIX}/${TF_SITEPACKAGES_REL})
  endif()
  message("TF_SITEPACKAGES: ${TF_SITEPACKAGES}")

endif(TF_WITH_PYTHON)

find_package(OpenGL REQUIRED)

add_definitions(-DCX3D_SUPPORT_OPENGL)

include_directories(${OPENGL_INCLUDE_DIR})

if(DEFINED ENV{JSON_INCLUDE_DIRS})
  set(JSON_INCLUDE_DIRS "$ENV{JSON_INCLUDE_DIRS}" CACHE STRING "")
else()
  set(JSON_INCLUDE_DIRS ${CMAKE_FIND_ROOT_PATH}/include)
endif()
message("JSON include: ${JSON_INCLUDE_DIRS}")


# process dependencies first
add_subdirectory(extern)

find_package(MagnumPlugins REQUIRED StbImageConverter)

add_subdirectory(include)
add_subdirectory(source)
add_subdirectory(wraps)

# CMake policies: enable MACOSX_RPATH by default
if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
# Don't treat imported targets with :: as files
if(POLICY CMP0028)
    cmake_policy(SET CMP0028 NEW)
endif()

find_package(Magnum REQUIRED Shaders GlfwApplication)

add_subdirectory(examples)
add_subdirectory(package)

set(
  TF_INFO_FILES
  README.md 
  COPYING.txt 
  COPYING.LESSER.txt
  LICENSE.txt 
  VERSION.txt 
)
install(FILES ${TF_INFO_FILES} DESTINATION ${CMAKE_INFO_PREFIX})
