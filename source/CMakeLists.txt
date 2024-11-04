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

add_compile_options(-DTF_STRICT)

add_subdirectory(mdcore)

# Resources for the tfMeshShaderProgram
corrade_add_resource(tfMeshShaderProgramRes tfMeshShaderProgram.res)

# Alias Magnum dependencies for convenient private linking, since export sets currently aren't available
add_library(TissueForge-magnum-interface INTERFACE)
target_link_libraries(
  TissueForge-magnum-interface INTERFACE 
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  MagnumGlfwApplication
  Magnum::SceneGraph
  Magnum::Primitives
  Magnum::MeshTools
  MagnumPlugins::StbImageConverter
  MagnumPlugins::StbTrueTypeFont
  MagnumExtras::Ui
  TgaImageConverter
)
if(TF_APPLE)
  target_link_libraries(TissueForge-magnum-interface INTERFACE MagnumWindowlessCglApplication)
elseif(TF_LINUX)
  target_link_libraries(TissueForge-magnum-interface INTERFACE MagnumWindowlessEglApplication)
elseif(TF_WINDOWS)
  target_link_libraries(TissueForge-magnum-interface INTERFACE MagnumWindowlessWglApplication)
endif()
add_library(TissueForge::MagnumLibs ALIAS TissueForge-magnum-interface)

################
# Source setup #
################

set(
  SRC
  tf_bind.cpp
  tf_metrics.cpp
  tf_parse.cpp
  tf_system.cpp
  tf_util.cpp
  tfError.cpp
  tfLogger.cpp
  tfSimulator.cpp
  tfUniverse.cpp

  io/tfThreeDFEdgeData.cpp
  io/tfThreeDFFaceData.cpp
  io/tfThreeDFRenderData.cpp
  io/tfThreeDFMeshData.cpp
  io/tfThreeDFStructure.cpp
  io/tfThreeDFVertexData.cpp
  io/generators/tfThreeDFAngleMeshGenerator.cpp
  io/generators/tfThreeDFBondMeshGenerator.cpp
  io/generators/tfThreeDFDihedralMeshGenerator.cpp
  io/generators/tfThreeDFMeshGenerator.cpp
  io/generators/tfThreeDFPCloudMeshGenerator.cpp
  io/tf_io.cpp
  io/tfThreeDFIO.cpp
  io/tfFIO.cpp
  io/tfIO.cpp

  event/tfEvent.cpp
  event/tfEventList.cpp
  event/tfParticleEvent.cpp
  event/tfParticleEventSingle.cpp
  event/tfParticleTimeEvent.cpp
  event/tfTimeEvent.cpp

  rendering/tfAngleRenderer.cpp
  rendering/tfAngleRenderer3D.cpp
  rendering/tfApplication.cpp
  rendering/tfArcBall.cpp
  rendering/tfArcBallInteractor.cpp
  rendering/tfArrowRenderer.cpp
  rendering/tfBondRenderer.cpp
  rendering/tfBondRenderer3D.cpp
  rendering/tfClipPlane.cpp
  rendering/tfColorMapper.cpp
  rendering/tfColorMaps.cpp
  rendering/tfDihedralRenderer.cpp
  rendering/tfDihedralRenderer3D.cpp
  rendering/tfEglInfo.cpp
  rendering/tfGlfwApplication.cpp
  rendering/tfGlfwWindow.cpp
  rendering/tfGlInfo.cpp
  rendering/tfImageConverters.cpp
  rendering/tfKeyEvent.cpp
  rendering/tfOrientationRenderer.cpp
  rendering/tfRenderer.cpp
  rendering/tfStyle.cpp
  rendering/tfSubRenderer.cpp
  rendering/tfUI.cpp
  rendering/tfUniverseRenderer.cpp
  rendering/tfWidgetRenderer.cpp
  rendering/tfWindow.cpp
  rendering/tfWindowlessApplication.cpp

  shaders/tfFlat3D.cpp
  shaders/tfParticleSphereShader.cpp
  shaders/tfPhong.cpp

  state/tfSpecies.cpp
  state/tfSpeciesList.cpp
  state/tfSpeciesValue.cpp
  state/tfStateVector.cpp

  types/tf_cast.cpp
  types/tfMagnum.cpp
)

set(
  HEADERS
  TissueForge_private.h
  
  tf_bind.h
  tf_debug.h
  tf_metrics.h
  tf_parse.h
  tf_system.h
  tf_test.h
  tf_util.h
  tfError.h
  tfLogger.h
  tfSimulator.h
  tfTaskScheduler.h
  tfThreadPool.h
  tfUniverse.h

  io/tfThreeDFEdgeData.h
  io/tfThreeDFFaceData.h
  io/tfThreeDFMeshData.h
  io/tfThreeDFRenderData.h
  io/tfThreeDFStructure.h
  io/tfThreeDFVertexData.h
  io/generators/tfThreeDFAngleMeshGenerator.h
  io/generators/tfThreeDFBondMeshGenerator.h
  io/generators/tfThreeDFDihedralMeshGenerator.h
  io/generators/tfThreeDFMeshGenerator.h
  io/generators/tfThreeDFPCloudMeshGenerator.h
  io/tfThreeDFIO.h
  io/tfFIO.h
  io/tfIO.h
  io/tf_io.h

  event/tfEvent.h
  event/tfEventList.h
  event/tfParticleEvent.h
  event/tfParticleEventSingle.h
  event/tfParticleTimeEvent.h
  event/tfTimeEvent.h

  rendering/tfAngleRenderer.h
  rendering/tfAngleRenderer3D.h
  rendering/tfApplication.h
  rendering/tfArcBall.h
  rendering/tfArcBallCamera.h
  rendering/tfArcBallInteractor.h
  rendering/tfArrowRenderer.h
  rendering/tfBondRenderer.h
  rendering/tfBondRenderer3D.h
  rendering/tfClipPlane.h
  rendering/tfColorMapper.h
  rendering/tfColorMaps.h
  rendering/tfDihedralRenderer.h
  rendering/tfDihedralRenderer3D.h
  rendering/tfEglInfo.h
  rendering/tfGlInfo.h
  rendering/tfGlfwApplication.h
  rendering/tfGlfwWindow.h
  rendering/tfImageConverters.h
  rendering/tfKeyEvent.h
  rendering/tfOrientationRenderer.h
  rendering/tfRenderer.h
  rendering/tfStyle.h
  rendering/tfSubRenderer.h
  rendering/tfUI.h
  rendering/tfUniverseRenderer.h
  rendering/tfWidgetRenderer.h
  rendering/tfWindow.h
  rendering/tfWindowless.h
  rendering/tfWindowlessApplication.h
  rendering/colormaps/colormaps.h

  shaders/tfFlat3D.h
  shaders/tfPhong.h
  shaders/tfParticleSphereShader.h

  state/tfSpecies.h
  state/tfSpeciesList.h
  state/tfSpeciesValue.h
  state/tfStateVector.h

  types/tf_cast.h
  types/tf_types.h
  types/tfMagnum.h
  types/tfMatrix.h
  types/tfMatrix3.h
  types/tfMatrix4.h
  types/tfQuaternion.h
  types/tfVector.h
  types/tfVector2.h
  types/tfVector3.h
  types/tfVector4.h
)

if(MSVC) 
  # Note to refactor tf_metrics.cpp
  set_source_files_properties(tf_metrics.cpp PROPERTIES COMPILE_OPTIONS "/bigobj")
endif()

# Python source

if(TF_WITH_PYTHON)

  list(
    APPEND SRC 
    langs/py/tf_py.cpp 

    langs/py/tf_bindPy.cpp
    langs/py/tf_systemPy.cpp 
    langs/py/tfBoundaryConditionsPy.cpp 
    langs/py/tfEventPy.cpp 
    langs/py/tfEventPyExecutor.cpp 
    langs/py/tfForcePy.cpp 
    langs/py/tfKeyEventPy.cpp 
    langs/py/tfParticleEventPy.cpp 
    langs/py/tfParticleEventSinglePy.cpp 
    langs/py/tfParticleTimeEventPy.cpp 
    langs/py/tfPotentialPy.cpp 
    langs/py/tfSimulatorPy.cpp 
    langs/py/tfTimeEventPy.cpp
  )

  list(
    APPEND HEADERS 
    langs/py/tf_py.h 

    langs/py/tf_bindPy.h
    langs/py/tf_systemPy.h 
    langs/py/tfBoundaryConditionsPy.h 
    langs/py/tfEventPy.h 
    langs/py/tfEventPyExecutor.h 
    langs/py/tfForcePy.h 
    langs/py/tfKeyEventPy.h 
    langs/py/tfParticleEventPy.h 
    langs/py/tfParticleEventSinglePy.h 
    langs/py/tfParticleTimeEventPy.h 
    langs/py/tfPotentialPy.h 
    langs/py/tfSimulatorPy.h 
    langs/py/tfTimeEventPy.h
  )

endif()

# CUDA source

if(TF_WITHCUDA)
  list(
    APPEND SRC 
    tf_cuda.cpp
    cuda/tfAngleConfig.cpp 
    cuda/tfBondConfig.cpp 
    cuda/tfEngineConfig.cpp 
    cuda/tfSimulatorConfig.cpp
  )
  list(
    APPEND HEADERS 
    tf_cuda.h
    cuda/tfAngleConfig.h 
    cuda/tfBondConfig.h 
    cuda/tfEngineConfig.h 
    cuda/tfSimulatorConfig.h
  )
endif()

set(HEADERS_PRIVATE ${HEADERS})
foreach(_PSHDR ${TISSUEFORGE_PUBLIC_HEADERS_STATIC})
  list(APPEND ${HEADERS} ${PROJECT_SOURCE_DIR}/include/${_PSHDR})
endforeach()

include(models/modeltree.cmake)
add_subdirectory(models)
TF_MODEL_TREE_INIT()
TF_MODEL_TREE_PROC(models)
set(SRC ${SRC} ${TF_MODEL_SRCS_LOCAL})

#######################
# Main object library #
#######################

# object library target: compiles the sources only once
add_library(
  TissueForge_obj
  OBJECT
  ${SRC}
  ${PROJECT_BINARY_DIR}/include/tf_config.h
  ${HEADERS}
  ${tfMeshShaderProgramRes}
)

target_compile_definitions(TissueForge_obj PRIVATE C_BUILDING_DLL)

set_target_properties(
  TissueForge_obj 
  PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
)

target_include_directories(
  TissueForge_obj PRIVATE
  ${ASSIMP_INCLUDE_DIRS}
  ${JSON_INCLUDE_DIRS}
)

target_include_directories(
  TissueForge_obj PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${TF_INCLUDE_DIR}> 
  $<INSTALL_INTERFACE:${TF_PRIVATE_INCLUDE_DIR}>
)

if(TF_WITH_PYTHON)
  target_include_directories(TissueForge_obj PUBLIC ${Python_INCLUDE_DIRS})
endif()

target_link_libraries(
  TissueForge_obj PRIVATE 
  Magnum::Application
  Magnum::Magnum
  Magnum::Shaders
  MagnumPlugins::StbImageConverter
  MagnumPlugins::StbTrueTypeFont
  MagnumExtras::Ui
  mdcore
)
target_link_libraries(
  TissueForge_obj PUBLIC 
  GLFW::GLFW 
  ${GLFW_LIBRARIES} 
  LibSBML::LibSBML
  assimp::assimp
  ${OPENGL_LIBRARIES}
)

add_dependencies(TissueForge_obj MagnumPlugins::StbImageConverter MagnumPlugins::StbTrueTypeFont MagnumExtras::Ui)

if(TF_WITHCUDA)
  target_link_libraries(
    TissueForge_obj PUBLIC 
    CUDA::cudart 
    CUDA::cuda_driver 
    CUDA::nvrtc
  )
endif()

# shared libraries need PIC
set_property(TARGET TissueForge_obj PROPERTY POSITION_INDEPENDENT_CODE 1)

#################
# Library setup #
#################

set(TF_LIB_TARGETS
  $<TARGET_OBJECTS:TissueForge_obj>
  $<TARGET_OBJECTS:mdcore>
  TissueForge.cpp
)
set(TF_LIBDEPS_PUBLIC
  GLFW::GLFW 
  ${GLFW_LIBRARIES} 
  LibSBML::LibSBML
  assimp::assimp
)
set(TF_LIBDEPS_PRIVATE
  mdcore
  TissueForge::MagnumLibs
)
set(TF_INCLUDES_PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${ASSIMP_INCLUDE_DIRS}
  ${JSON_INCLUDE_DIRS}
  $<INSTALL_INTERFACE:${TF_INCLUDE_DIR}> 
  $<INSTALL_INTERFACE:${TF_PRIVATE_INCLUDE_DIR}>
)

if(TF_WITH_PYTHON)
  list(APPEND TF_INCLUDES_PUBLIC ${Python_INCLUDE_DIRS})
  list(APPEND TF_LIBDEPS_PUBLIC ${Python_LIBRARY_RELEASE})
endif()

if(TF_WITHCUDA)
  list(APPEND TF_INCLUDES_PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
  list(APPEND TF_LIBDEPS_PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
endif()

##################
# Static library #
##################

# shared and static libraries built from the same object files
add_library(TissueForge_static STATIC ${TF_LIB_TARGETS})

target_compile_definitions(TissueForge_static PRIVATE C_BUILDING_DLL)
set_target_properties(TissueForge_static 
  PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  OUTPUT_NAME tissue-forge-static
)
target_include_directories(TissueForge_static PUBLIC ${TF_INCLUDES_PUBLIC})
target_link_libraries(TissueForge_static PRIVATE ${TF_LIBDEPS_PRIVATE})
target_link_libraries(TissueForge_static PUBLIC ${TF_LIBDEPS_PUBLIC})

if(TF_WITHCUDA)
  set_target_properties(TissueForge_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(TF_CUDA_DEBUG)
    target_compile_options(TissueForge_static PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
  endif()
endif()

if(TF_WITH_PYTHON)
  # weird bug with CMake / clang on windows, need to add link dir here
  target_link_directories(TissueForge_static PUBLIC ${Python_LIBRARY_DIRS})
endif()


##################
# Shared library #
##################

# shared and static libraries built from the same object files
add_library(TissueForge_shared SHARED ${TF_LIB_TARGETS})

target_compile_definitions(TissueForge_shared PRIVATE C_BUILDING_DLL)
set_target_properties(TissueForge_shared 
  PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  OUTPUT_NAME tissue-forge
)
target_include_directories(TissueForge_shared PUBLIC ${TF_INCLUDES_PUBLIC})
target_link_libraries(TissueForge_shared PRIVATE ${TF_LIBDEPS_PRIVATE})
target_link_libraries(TissueForge_shared PUBLIC ${TF_LIBDEPS_PUBLIC})

if(TF_WITHCUDA)
  set_target_properties(TissueForge_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  if(TF_CUDA_DEBUG)
    target_compile_options(TissueForge_shared PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
  endif()
endif()

if(TF_WITH_PYTHON)
  # weird bug with CMake / clang on windows, need to add link dir here
  target_link_directories(TissueForge_shared PUBLIC ${Python_LIBRARY_DIRS})
endif()

if(APPLE)
  if(ARG_SHARED)
    # Suppress CMake >= 3.0 warning for shared libraries
    set_target_properties(TissueForge_shared PROPERTIES MACOSX_RPATH ON)
  endif()
endif()


# Generate aliases

add_library(TissueForge::Static ALIAS TissueForge_static)
add_library(TissueForge::Shared ALIAS TissueForge_shared)

###############################
# Install targets and exports #
###############################

install(
  TARGETS TissueForge_shared 
  EXPORT TissueForge_shared_targets 
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" 
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" 
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

install(
  EXPORT TissueForge_shared_targets 
  FILE TissueForgeTargets.cmake 
  NAMESPACE TissueForge:: 
  DESTINATION ${TF_SHARE_DIR} 
  EXPORT_LINK_INTERFACE_LIBRARIES 
  COMPONENT Shared
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/TissueForgeConfig.cmake.in 
  ${CMAKE_CURRENT_BINARY_DIR}/TissueForgeConfig.cmake 
  INSTALL_DESTINATION ${TF_SHARE_DIR} 
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/TissueForgeConfig.cmake DESTINATION ${TF_SHARE_DIR})

foreach(_PRHDR ${HEADERS_PRIVATE})
  cmake_path(GET _PRHDR PARENT_PATH _PRHDR_PARENT)
  install(FILES ${_PRHDR} DESTINATION ${TF_PRIVATE_INCLUDE_DIR}/${_PRHDR_PARENT})
endforeach()
