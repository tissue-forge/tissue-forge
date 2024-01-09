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

cmake_minimum_required (VERSION 3.20.3)
project(TissueForgeTests)
include(CTest)

option(TF_WITH_C        "Test C language interface for Tissue Forge."      ON)
option(TF_WITH_CPP      "Test C++ language interface for Tissue Forge."    ON)
option(TF_WITH_PYTHON   "Test Python language interface for Tissue Forge." ON)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

if(TF_WITH_C)
    add_subdirectory(C)
endif()

if(TF_WITH_CPP)
    add_subdirectory(cpp)
endif()

if(TF_WITH_PYTHON)
    add_subdirectory(python)
endif()
