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

find_package(Python 3.8.0...3.11.99 REQUIRED COMPONENTS Interpreter)

macro(tfPyTest_Test src_name src_path)
    add_test(
        NAME ${src_name} 
        COMMAND Python::Interpreter -m pytest ${src_path}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endmacro()

file(GLOB tfPyTest_SRCS tfPyTest_*.py)
foreach(tfPyTest_SRC ${tfPyTest_SRCS})
    cmake_path(GET tfPyTest_SRC FILENAME tfPyTest_SRC_FN)
    cmake_path(GET tfPyTest_SRC_FN STEM LAST_ONLY tfPyTest_SRC_NAME)
    tfPyTest_Test(${tfPyTest_SRC_NAME} ${tfPyTest_SRC})
endforeach()
