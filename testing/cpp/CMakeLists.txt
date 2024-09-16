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

# Cross-compiling needs some flexibility concerning whether to rely on CMAKE_FIND_ROOT_PATH, so provide flexibility here. 
if(DEFINED ENV{JSON_INCLUDE_DIRS})
  set(JSON_INCLUDE_DIRS "$ENV{JSON_INCLUDE_DIRS}" CACHE STRING "")
endif()
if(NOT DEFINED JSON_INCLUDE_DIRS) 
  set(JSON_INCLUDE_DIRS ${CMAKE_FIND_ROOT_PATH}/include)
endif()
message("JSON include: ${JSON_INCLUDE_DIRS}")

find_package(CUDAToolkit)
find_package(assimp REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
if(UNIX OR CYGWIN)
    find_package(sbml-static REQUIRED)
else()
  if(MINGW)
    find_package(sbml-static REQUIRED)
  else()
    find_package(libsbml-static REQUIRED)
  endif()
endif()
find_package(TissueForge REQUIRED COMPONENTS Shared)

set(tfTest_LIBS TissueForge::TissueForge_shared)
if(UNIX)
    list(APPEND tfTest_LIBS m)
endif()

set(
  tfTest_INCLUDES 
  ${TISSUEFORGE_INCLUDE_DIRS} 
  ${ASSIMP_INCLUDE_DIRS}
  ${JSON_INCLUDE_DIRS}
)

include_directories(${OPENGL_INCLUDE_DIR})

if(${TISSUEFORGE_HAS_CUDA})
  list(APPEND tfTest_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
endif()

macro(tfTest_Build src_name)
    add_executable(${src_name} ${src_name}.cpp)
    target_compile_definitions(${src_name} PRIVATE TFTEST_LOG)
    target_link_libraries(${src_name} ${tfTest_LIBS})
    target_include_directories(${src_name} PUBLIC ${tfTest_INCLUDES})
endmacro()

file(GLOB tfTest_SRCS tfTest_*.cpp)
foreach(tfTest_SRC ${tfTest_SRCS})
    cmake_path(GET tfTest_SRC FILENAME tfTest_SRC_FN)
    cmake_path(GET tfTest_SRC_FN STEM LAST_ONLY tfTest_SRC_NAME)
    tfTest_Build(${tfTest_SRC_NAME})
    add_test(NAME ${tfTest_SRC_NAME} COMMAND ${tfTest_SRC_NAME})
endforeach()
