#*******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022, 2023 T.J. Sego
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

find_package(CUDAToolkit)
find_package(assimp REQUIRED)
find_package(glfw3 REQUIRED)
if(UNIX OR CYGWIN)
    find_package(sbml-static REQUIRED)
else()
  if(MINGW)
    find_package(sbml-static REQUIRED)
  else()
    find_package(libsbml-static REQUIRED)
  endif()
endif()
find_package(TissueForge REQUIRED COMPONENTS CLib)

set(tfCTest_LIBS TissueForge::TissueForge_c)
if(UNIX)
    list(APPEND tfCTest_LIBS m)
endif()

set(tfCTest_INCLUDES ${TISSUEFORGE_C_INCLUDE_DIRS})

macro(tfCTest_Build src_name)
    add_executable(${src_name} ${src_name}.c)
    target_link_libraries(${src_name} ${tfCTest_LIBS})
    target_include_directories(${src_name} PUBLIC ${tfCTest_INCLUDES})
endmacro()

file(GLOB tfCTest_SRCS tfCTest_*.c)
foreach(tfCTest_SRC ${tfCTest_SRCS})
    cmake_path(GET tfCTest_SRC FILENAME tfCTest_SRC_FN)
    cmake_path(GET tfCTest_SRC_FN STEM LAST_ONLY tfCTest_SRC_NAME)
    tfCTest_Build(${tfCTest_SRC_NAME})
    add_test(NAME ${tfCTest_SRC_NAME} COMMAND ${tfCTest_SRC_NAME})
endforeach()
