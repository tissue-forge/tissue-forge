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

set(
  SOURCES 
  TissueForge_c.cpp 
  TissueForge_c_private.cpp 

  tfC_bind.cpp 
  tfC_event.cpp 
  tfC_io.cpp 
  tfC_system.cpp 
  tfC_util.cpp 
  tfCBond.cpp 
  tfCBoundaryConditions.cpp 
  tfCClipPlane.cpp 
  tfCCluster.cpp 
  tfCError.cpp 
  tfCFlux.cpp 
  tfCForce.cpp 
  tfCLogger.cpp 
  tfCParticle.cpp 
  tfCPotential.cpp 
  tfCSimulator.cpp 
  tfCSpecies.cpp 
  tfCStateVector.cpp 
  tfCStyle.cpp 
  tfCUniverse.cpp 
)

set(
  HEADERS
  TissueForge_c.h 
  TissueForge_c_private.h 
  tf_port_c.h

  tfC_bind.h 
  tfC_event.h 
  tfC_io.h 
  tfC_system.h 
  tfC_util.h 
  tfCBond.h 
  tfCBoundaryConditions.h 
  tfCClipPlane.h 
  tfCCluster.h 
  tfCError.h 
  tfCFlux.h 
  tfCForce.h 
  tfCLogger.h 
  tfCParticle.h 
  tfCPotential.h 
  tfCSimulator.h 
  tfCSpecies.h 
  tfCStateVector.h 
  tfCStyle.h 
  tfCUniverse.h 
)

if(TF_WITHCUDA) 
  list(APPEND SOURCES tfC_cuda.cpp)
  list(APPEND HEADERS tfC_cuda.h)
endif()

add_subdirectory(models)
TF_MODEL_TREE_INIT()
TF_MODEL_TREE_PROC(models)
set(SOURCES ${SOURCES} ${TF_MODEL_SRCS_LOCAL})

add_library(TissueForge_c SHARED ${SOURCES} ${HEADERS})

target_compile_definitions(TissueForge_c PRIVATE C_BUILDING_DLL)

set_target_properties(
  TissueForge_c PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1 
  POSITION_INDEPENDENT_CODE 1 
  OUTPUT_NAME tissue-forge-c
)

target_include_directories(
  TissueForge_c PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${TF_WRAPS_C_INSTALL_PREFIX}>
)

if(TF_WITHCUDA)
  target_include_directories(
    TissueForge_c PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
  )
  set_target_properties(
    TissueForge_c PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

target_link_libraries(TissueForge_c PUBLIC TissueForge::Shared)
target_link_libraries(
  TissueForge_c PRIVATE 
  TissueForge_obj 
  mdcore 
  TissueForge::MagnumLibs
)

if(TF_WITHCUDA)
  target_link_libraries(
    TissueForge_c PUBLIC 
    CUDA::cudart 
    CUDA::cuda_driver 
    CUDA::nvrtc
  )
endif()

add_library(TissueForge::CLib ALIAS TissueForge_c)

install(
  TARGETS TissueForge_c 
  EXPORT TissueForge_c_targets 
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" 
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" 
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(FILES ${HEADERS} DESTINATION ${TF_WRAPS_C_INSTALL_PREFIX})
install(
  EXPORT TissueForge_c_targets 
  FILE TissueForge_CTargets.cmake 
  NAMESPACE TissueForge:: 
  DESTINATION ${TF_SHARE_DIR} 
  EXPORT_LINK_INTERFACE_LIBRARIES 
  COMPONENT CLib
)
