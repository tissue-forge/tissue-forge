#*******************************************************************************
# This file is part of Tissue Forge.
# Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
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

TF_MODEL_TREE_INIT()

TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tf_mesh_bind.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tf_mesh_create.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfBody.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfMeshObj.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfMeshQuality.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfMeshLogger.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfMesh.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfMeshRenderer.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfMeshSolver.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfStructure.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfSurface.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfVertex.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tf_mesh_metrics.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfBodyForce.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfNormalStress.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfSurfaceAreaConstraint.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfSurfaceTraction.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfVolumeConstraint.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfEdgeTension.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfAdhesion.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfFlatSurfaceConstraint.cpp)

TF_MODEL_TREE_HDR(vertex/solver/tf_mesh.h)
TF_MODEL_TREE_HDR(vertex/solver/tf_mesh_bind.h)
TF_MODEL_TREE_HDR(vertex/solver/tf_mesh_create.h)
TF_MODEL_TREE_HDR(vertex/solver/tfBody.h)
TF_MODEL_TREE_HDR(vertex/solver/tfMesh.h)
TF_MODEL_TREE_HDR(vertex/solver/tfMeshQuality.h)
TF_MODEL_TREE_HDR(vertex/solver/tfMeshLogger.h)
TF_MODEL_TREE_HDR(vertex/solver/tfMeshObj.h)
TF_MODEL_TREE_HDR(vertex/solver/tfMeshRenderer.h)
TF_MODEL_TREE_HDR(vertex/solver/tfMeshSolver.h)
TF_MODEL_TREE_HDR(vertex/solver/tfStructure.h)
TF_MODEL_TREE_HDR(vertex/solver/tfSurface.h)
TF_MODEL_TREE_HDR(vertex/solver/tfVertex.h)
TF_MODEL_TREE_HDR(vertex/solver/tf_mesh_metrics.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfBodyForce.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfNormalStress.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfSurfaceAreaConstraint.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfSurfaceTraction.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfVolumeConstraint.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfEdgeTension.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfAdhesion.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfFlatSurfaceConstraint.h)
