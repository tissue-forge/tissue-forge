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

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit)
find_package(assimp REQUIRED)
find_package(glfw3 REQUIRED)
find_package(libsbml-static REQUIRED)
find_package(TissueForge REQUIRED COMPONENTS Shared)

set(tfExamples_LIBS TissueForge::TissueForge_shared)
if(UNIX)
    list(APPEND tfExamples_LIBS m)
endif()

set(tfExamples_INCLUDES ${TISSUEFORGE_INCLUDE_DIRS})
