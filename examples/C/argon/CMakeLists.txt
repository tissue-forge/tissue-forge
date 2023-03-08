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

cmake_minimum_required(VERSION 3.20.3)
project(argon)

set(TF_INSTALL_ROOT "" CACHE PATH "Root directory of Tissue Forge installation")
set(TF_APPNAME argonC)

include(${CMAKE_CURRENT_SOURCE_DIR}/../TissueForgeExamples.cmake)

add_executable(${TF_APPNAME}App argon.c)

target_include_directories(${TF_APPNAME}App PRIVATE ${tfExamples_INCLUDES})
target_link_libraries(${TF_APPNAME}App PRIVATE ${tfExamples_LIBS})
target_link_directories(${TF_APPNAME}App PRIVATE ${TF_INSTALL_ROOT}/lib)

install(TARGETS ${TF_APPNAME}App DESTINATION ${TF_INSTALL_ROOT}/bin)
