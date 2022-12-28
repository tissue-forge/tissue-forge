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

TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfC_mesh_bind.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfC_mesh_create.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfCBody.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfCMesh.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfCMeshLogger.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfCMeshObj.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfCMeshQuality.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfCMeshSolver.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfCSurface.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfCVertex.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/tfCVertexSolverFIO.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfCAdhesion.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfCBodyForce.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfCConvexPolygonConstraint.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfCEdgeTension.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfCFlatSurfaceConstraint.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfCNormalStress.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfCSurfaceAreaConstraint.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfCSurfaceTraction.cpp)
TF_MODEL_TREE_SRC(${CMAKE_CURRENT_SOURCE_DIR}/actors/tfCVolumeConstraint.cpp)

TF_MODEL_TREE_HDR(vertex/solver/tfC_mesh_bind.h)
TF_MODEL_TREE_HDR(vertex/solver/tfC_mesh_create.h)
TF_MODEL_TREE_HDR(vertex/solver/tfCBody.h)
TF_MODEL_TREE_HDR(vertex/solver/tfCMesh.h)
TF_MODEL_TREE_HDR(vertex/solver/tfCMeshLogger.h)
TF_MODEL_TREE_HDR(vertex/solver/tfCMeshObj.h)
TF_MODEL_TREE_HDR(vertex/solver/tfCMeshQuality.h)
TF_MODEL_TREE_HDR(vertex/solver/tfCMeshSolver.h)
TF_MODEL_TREE_HDR(vertex/solver/tfCSurface.h)
TF_MODEL_TREE_HDR(vertex/solver/tfCVertex.h)
TF_MODEL_TREE_HDR(vertex/solver/tfCVertexSolverFIO.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfC_actors.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfCAdhesion.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfCBodyForce.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfCConvexPolygonConstraint.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfCEdgeTension.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfCFlatSurfaceConstraint.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfCNormalStress.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfCSurfaceAreaConstraint.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfCSurfaceTraction.h)
TF_MODEL_TREE_HDR(vertex/solver/actors/tfCVolumeConstraint.h)
