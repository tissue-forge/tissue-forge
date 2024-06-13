#*******************************************************************************
# This file is part of mdcore.
# Coypright (c) 2010 Pedro Gonnet (gonnet@maths.ox.ac.uk)
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
#*******************************************************************************

include_directories(${FFTW_INCLUDES})

link_libraries(${FFTW_LIBRARIES})

set(
  PUBLIC_HEADERS
  "${MDCORE_SOURCE_DIR}/include/cycle.h"
  "${MDCORE_SOURCE_DIR}/include/mdcore_config.h"
  "${MDCORE_SOURCE_DIR}/include/tf_cycle.h"
  "${MDCORE_SOURCE_DIR}/include/tf_errs.h"
  "${MDCORE_SOURCE_DIR}/include/tf_fptype.h"
  "${MDCORE_SOURCE_DIR}/include/tf_lock.h"
  "${MDCORE_SOURCE_DIR}/include/tf_platform.h"
  "${MDCORE_SOURCE_DIR}/include/tfAngle.h"
  "${MDCORE_SOURCE_DIR}/include/tfBond.h"
  "${MDCORE_SOURCE_DIR}/include/tfBoundaryConditions.h"
  "${MDCORE_SOURCE_DIR}/include/tfCluster.h"
  "${MDCORE_SOURCE_DIR}/include/tfDihedral.h"
  "${MDCORE_SOURCE_DIR}/include/tfDPDPotential.h"
  "${MDCORE_SOURCE_DIR}/include/tfEngine.h"
  "${MDCORE_SOURCE_DIR}/include/tfExclusion.h"
  "${MDCORE_SOURCE_DIR}/include/tfFlux.h"
  "${MDCORE_SOURCE_DIR}/include/tfForce.h"
  "${MDCORE_SOURCE_DIR}/include/tfParticle.h"
  "${MDCORE_SOURCE_DIR}/include/tfParticleList.h"
  "${MDCORE_SOURCE_DIR}/include/tfParticleTypeList.h"
  "${MDCORE_SOURCE_DIR}/include/tfPotential.h"
  "${MDCORE_SOURCE_DIR}/include/tfQueue.h"
  "${MDCORE_SOURCE_DIR}/include/tfRigid.h"
  "${MDCORE_SOURCE_DIR}/include/tfRunner.h"
  "${MDCORE_SOURCE_DIR}/include/tfSecreteUptake.h"
  "${MDCORE_SOURCE_DIR}/include/tfSpace.h"
  "${MDCORE_SOURCE_DIR}/include/tfSpace_cell.h"
  "${MDCORE_SOURCE_DIR}/include/tfTask.h"
)

set(
  PRIVATE_HEADERS
  tf_boundary_eval.h
  tf_dpd_eval.h
  tf_engine_advance.h
  tf_flux_eval.h
  tf_mdcore_io.h
  tf_potential_eval.h
  tf_smoothing_kernel.h
  tfBtree.h
  tfSubEngine.h
)

# List required headers

# Both precisions have the same sources.
set(
  SOURCES
  tf_engine_advance.cpp
  tf_engine_bonded.cpp
  tf_engine_rigid.cpp
  tf_runner_dopair.cpp
  tf_runner_dosort.cpp
  tf_runner_verlet.cpp
  tfAngle.cpp
  tfBond.cpp
  tfBoundaryConditions.cpp
  tfCluster.cpp
  tfDihedral.cpp
  tfDPDPotential.cpp
  tfExclusion.cpp
  tfEngine.cpp
  tfFlux.cpp
  tfForce.cpp
  tfParticle.cpp
  tfParticleList.cpp
  tfParticleTypeList.cpp
  tfPotential.cpp
  tfQueue.cpp
  tfRigid.cpp
  tfRunner.cpp
  tfSecreteUptake.cpp
  tfSpace.cpp
  tfSpace_cell.cpp
  tfSubEngine.cpp
  tfTask.cpp
)

if(MDCORE_DOUBLE)
  set(MDCORE_COMPILEDEFS_PUBLIC MDCORE_DOUBLE FPTYPE_DOUBLE)
else()
  set(MDCORE_COMPILEDEFS_PUBLIC MDCORE_SINGLE)
endif()

if(TF_HAVEPTHREADS)
  list(APPEND MDCORE_COMPILEDEFS_PUBLIC PTHREAD_LOCK)
endif()
set(MDCORE_COMPILEDEFS_PRIVATE C_BUILDING_DLL)
set(
  MDCORE_LIBS   
  LibSBML::LibSBML
  Threads::Threads
  Magnum::Magnum
)

if(MDCORE_USE_MPI)
  set(SOURCES ${SOURCES} engine_exchange.cpp)
endif()

# CUDA sources

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND AND TF_WITHCUDA)
  set(MDCORE_HAVECUDA ON)
else()
  set(MDCORE_HAVECUDA OFF)
endif()

if(MDCORE_HAVECUDA)
  list(APPEND MDCORE_COMPILEDEFS_PRIVATE HAVE_CUDA)

  list(
    APPEND PUBLIC_HEADERS 
    "${MDCORE_SOURCE_DIR}/src/tfAngle_cuda.h"
    "${MDCORE_SOURCE_DIR}/src/tfBond_cuda.h"
    "${MDCORE_SOURCE_DIR}/src/tfBoundaryConditions_cuda.h" 
    "${MDCORE_SOURCE_DIR}/src/tfFlux_cuda.h"
    "${MDCORE_SOURCE_DIR}/src/tfParticle_cuda.h" 
    "${MDCORE_SOURCE_DIR}/src/tfPotential_cuda.h" 
    "${MDCORE_SOURCE_DIR}/src/tfRunner_cuda.h" 
  )
  list(
    APPEND SOURCES
    tfBond_cuda.cu 
    tfBoundaryConditions_cuda.cu  
    tfEngine_cuda.cu 
    tfFlux_cuda.cu 
    tfParticle_cuda.cu 
    tfPotential_cuda.cu 
    tfRunner_cuda.cu 
  )
endif()

set(
  MDCORE_INCLUDE_DIRS 
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> 
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source> 
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${MDCORE_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${MDCORE_SOURCE_DIR}/src> 
  $<BUILD_INTERFACE:${MDCORE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${TF_PRIVATE_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${TF_PRIVATE_INCLUDE_DIR}/mdcore>
  $<INSTALL_INTERFACE:${TF_PRIVATE_INCLUDE_DIR}/mdcore/private>
  ${Python_INCLUDE_DIRS}
)

if(MDCORE_DOUBLE)

  set(MDCORE_BUILD_SINGLE NO)
  SET(MDCORE_PRECISION_LABEL "double")
  add_definitions(-DMDCORE_DOUBLE)

else()
  
  set(MDCORE_BUILD_SINGLE YES)
  SET(MDCORE_PRECISION_LABEL "single")
  add_definitions(-DMDCORE_SINGLE)

endif()

set(MDCORE_PRECISION_FILE "mdcore_${MDCORE_PRECISION_LABEL}.h")
set(MDCORE_CONFIG_FILE "mdcore_${MDCORE_PRECISION_LABEL}_config.h")

configure_file("${MDCORE_SOURCE_DIR}/include/mdcore.h.in" "${MDCORE_BINARY_DIR}/include/${MDCORE_PRECISION_FILE}")
configure_file("${MDCORE_SOURCE_DIR}/include/mdcore_config.h.in" "${MDCORE_BINARY_DIR}/include/${MDCORE_CONFIG_FILE}")
set(
  PRECISION_HEADERS
  "${MDCORE_BINARY_DIR}/include/${MDCORE_PRECISION_FILE}"
  "${MDCORE_BINARY_DIR}/include/${MDCORE_CONFIG_FILE}"
)

if(MDCORE_HAVECUDA)
  list(
    APPEND MDCORE_LIBS 
    CUDA::cudart 
    CUDA::cuda_driver 
    CUDA::nvrtc
  )
endif()

add_library(mdcore OBJECT ${SOURCES} ${PUBLIC_HEADERS} ${PRIVATE_HEADERS} ${PRECISION_HEADERS})

target_compile_definitions(mdcore PUBLIC ${MDCORE_COMPILEDEFS_PUBLIC})
target_compile_definitions(mdcore PRIVATE ${MDCORE_COMPILEDEFS_PRIVATE})

set_target_properties(
  mdcore 
  PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1 
  POSITION_INDEPENDENT_CODE ON 
)
if(MDCORE_HAVECUDA)
  set_target_properties(
    mdcore 
    PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
  )
  if(TF_CUDA_DEBUG) 
    target_compile_options(mdcore PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
  endif()
endif()

target_include_directories(mdcore PUBLIC ${MDCORE_INCLUDE_DIRS} ${Python_INCLUDE_DIRS})
target_link_libraries(mdcore PUBLIC ${MDCORE_LIBS})

install(FILES ${PRECISION_HEADERS} DESTINATION ${TF_PRIVATE_INCLUDE_DIR}/mdcore)
install(FILES ${PUBLIC_HEADERS} DESTINATION ${TF_PRIVATE_INCLUDE_DIR}/mdcore)
install(FILES ${PRIVATE_HEADERS} DESTINATION ${TF_PRIVATE_INCLUDE_DIR}/mdcore/private)
