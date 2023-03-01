#*******************************************************************************
# This file is part of mdcore.
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
#*******************************************************************************

if(TF_FPTYPE_SINGLE) 
  set(MDCORE_DOUBLE NO)
else() 
  set(MDCORE_DOUBLE YES)
endif()

option(MDCORE_SHARED "build shared library (static library is alwasy built)" NO)

option(MDCORE_USE_MPI "build the MPI version" NO)

option(MDCORE_USE_FFTW3 "use FFTW" NO)

option(MDCORE_USE_OPENMP "used OpenMP (only available with GCC)" NO)

option(SANITIZE "build with -fsanitize=address" NO)

if(SANITIZE)
  add_compile_options(-fsanitize=address)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
endif()

# define dirs relative to root of mdcore project so it can be
# used as a sub-project
set(MDCORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MDCORE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})


# Add our module finders to cmake  |Demos/StaticBoundaryDemo/CMakeLists.txt:13:     ${GLUT_LIBRARIES}
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(MDCORE_USE_FFTW3)
  find_package(FFTW REQUIRED)
  message("FFTW_LIBRARIES: ${FFTW_LIBRARIES}")
  message("FFTW_INCLUDES: ${FFTW_INCLUDES}")
endif()


add_subdirectory(src)
