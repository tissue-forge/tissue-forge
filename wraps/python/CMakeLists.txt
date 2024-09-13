#*******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022-2024 T.J. Sego
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

find_package(SWIG 4.0 REQUIRED COMPONENTS python)
include(UseSWIG)

set(SWIG_COMPILEDEFS_PY )

# Power through int size mess until properly resolved 
# See https://github.com/swig/swig/issues/1850
include(CheckTypeSize)
check_type_size("long" SIZE_LONG)
message(STATUS "Got size of long: ${SIZE_LONG} (${HAVE_SIZE_LONG})")
if(NOT TF_APPLE)
  if(${SIZE_LONG} STREQUAL 8)
    list(APPEND SWIG_COMPILEDEFS_PY SWIGWORDSIZE64)
  else()
    list(APPEND SWIG_COMPILEDEFS_PY SWIGWORDSIZE32)
  endif()
endif()

if(TF_APPLE)
  list(APPEND SWIG_COMPILEDEFS_PY CORRADE_TARGET_APPLE)
endif()

if(MSVC)
  add_definitions(/bigobj)
endif()

set(TissueForge_py_LIBS
  mdcore

  glfw 
  ${GLFW_LIBRARIES} 
  LibSBML::LibSBML 
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  MagnumGlfwApplication
  Magnum::SceneGraph
  Magnum::Primitives
  Magnum::MeshTools
  MagnumPlugins::StbImageConverter
  TgaImageConverter
  assimp::assimp
)

if(TF_WITHCUDA)
  list(APPEND TissueForge_py_LIBS 
    CUDA::cudart 
    CUDA::cuda_driver 
    CUDA::nvrtc
  )
endif()

if(TF_APPLE)
  list(APPEND TissueForge_py_LIBS MagnumWindowlessCglApplication)
elseif(TF_LINUX)
  list(APPEND TissueForge_py_LIBS MagnumWindowlessEglApplication)
elseif(TF_WINDOWS)
  list(APPEND TissueForge_py_LIBS MagnumWindowlessWglApplication)
endif()

message(STATUS "SWIG_COMPILEDEFS_PY=${SWIG_COMPILEDEFS_PY}")
message(STATUS "seting python output dir to ${TF_PYPKG_DIR}")

macro(TF_BUILD_PY_LIB main_source target_name output_py_name output_subdir)

  message(STATUS "Processing python module: ${output_py_name} -> ${target_name} (${main_source})")

  set_source_files_properties(${main_source} PROPERTIES CPLUSPLUS ON SWIG_FLAGS "-keyword;-py3;-doxygen")
  set_property(SOURCE ${main_source} PROPERTY COMPILE_DEFINITIONS ${SWIG_COMPILEDEFS_PY})

  swig_add_library(${target_name} TYPE SHARED LANGUAGE python SOURCES ${main_source} $<TARGET_OBJECTS:TissueForge_obj>)

  target_compile_definitions(${target_name} PRIVATE C_BUILDING_DLL)

  set_target_properties(${target_name} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN 1
    SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )

  # copy the Tissue Forge py lib to the package dir
  set_target_properties(${target_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY          ${TF_PYPKG_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG    ${TF_PYPKG_DIR}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE  ${TF_PYPKG_DIR}

    LIBRARY_OUTPUT_DIRECTORY          ${TF_PYPKG_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG    ${TF_PYPKG_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE  ${TF_PYPKG_DIR}

    RUNTIME_OUTPUT_DIRECTORY          ${TF_PYPKG_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG    ${TF_PYPKG_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE  ${TF_PYPKG_DIR}
  )


  target_include_directories(${target_name} PRIVATE
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/mdcore/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source/mdcore/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source/mdcore/src>
    ${Python_INCLUDE_DIRS}
    $<INSTALL_INTERFACE:${TF_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:${TF_INCLUDE_DIR}/private/mdcore>
    $<INSTALL_INTERFACE:${TF_INCLUDE_DIR}/private/mdcore/private>
  )

  if(TF_WITHCUDA)
    target_include_directories(${target_name} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()

  swig_link_libraries(${target_name} PRIVATE ${TissueForge_py_LIBS})

  if(WIN32 OR CYGWIN OR (UNIX AND NOT APPLE))

    # Link against the Python shared library on Windows
    swig_link_libraries(${target_name} PRIVATE ${Python_LIBRARIES})

  elseif(APPLE)
    # It's quite common to have multiple copies of the same Python version
    # installed on one's system. E.g.: one copy from the OS and another copy
    # that's statically linked into an application like Blender or Maya.
    # If we link our plugin library against the OS Python here and import it
    # into Blender or Maya later on, this will cause segfaults when multiple
    # conflicting Python instances are active at the same time (even when they
    # are of the same version).

    # Windows is not affected by this issue since it handles DLL imports
    # differently. The solution for Linux and Mac OS is simple: we just don't
    # link against the Python library. The resulting shared library will have
    # missing symbols, but that's perfectly fine -- they will be resolved at
    # import time.

    target_link_libraries(${target_name} PRIVATE "-undefined dynamic_lookup")

    if(ARG_SHARED)
      # Suppress CMake >= 3.0 warning for shared libraries
      set_target_properties(${target_name} PROPERTIES MACOSX_RPATH ON)
    endif()
  endif()

  # Handle rpath
  if(UNIX)
    file(RELATIVE_PATH _RPATH2PYLIBS ${TF_SITEPACKAGES}/${output_subdir} ${Python_LIBRARY_DIRS})
    file(RELATIVE_PATH _RPATH2TFLIBS ${TF_SITEPACKAGES}/${output_subdir} ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
    file(RELATIVE_PATH _RPATH2TFENV  ${TF_SITEPACKAGES}/${output_subdir} ${CMAKE_INSTALL_PREFIX})
    set_target_properties(${target_name} PROPERTIES INSTALL_RPATH "${RPATH_PREFIX};${RPATH_PREFIX}/${_RPATH2PYLIBS};${RPATH_PREFIX}/${_RPATH2TFLIBS};${RPATH_PREFIX}/${_RPATH2TFENV}")
  endif()

  set_target_properties(
    ${target_name} 
    PROPERTIES 
    PREFIX ""
    OUTPUT_NAME "_${output_py_name}"
  )

  if(TF_WINDOWS)
    set_target_properties(${target_name} PROPERTIES SUFFIX ".pyd")
  else()
    set_target_properties(${target_name} PROPERTIES SUFFIX ".so")
  endif()

  install(TARGETS ${target_name} DESTINATION ${TF_SITEPACKAGES}/${output_subdir})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${output_py_name}.py DESTINATION ${TF_SITEPACKAGES}/${output_subdir})

endmacro()

TF_BUILD_PY_LIB(tissue_forge.i TissueForge_py tissue_forge tissue_forge)

add_library(TissueForge::PythonLib ALIAS TissueForge_py)

file(COPY __init__.py DESTINATION ${TF_PYPKG_DIR})
file(COPY lattice.py DESTINATION ${TF_PYPKG_DIR})
file(COPY jwidget.py DESTINATION ${TF_PYPKG_DIR})
file(COPY particle_type.py DESTINATION ${TF_PYPKG_DIR})
install(
  FILES 
  __init__.py 
  lattice.py 
  jwidget.py
  particle_type.py 
  bind.py 
  metrics.py 
  system.py 
  util.py 
  DESTINATION 
  ${TF_SITEPACKAGES}/tissue_forge
)
if(TF_WITHCUDA)
  install(FILES cuda.py DESTINATION ${TF_SITEPACKAGES}/tissue_forge)
endif()

add_subdirectory(event)
add_subdirectory(io)
add_subdirectory(mdcore)
add_subdirectory(models)
add_subdirectory(notebook)
add_subdirectory(rendering)
add_subdirectory(state)
add_subdirectory(types)
