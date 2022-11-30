# ******************************************************************************
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
# ******************************************************************************

from . import bind
from .mesh_types import BodyTypeSpec, SurfaceTypeSpec

from tissue_forge.tissue_forge import _vertex_solver_Body as Body
from tissue_forge.tissue_forge import _vertex_solver_BodyType as BodyType
from tissue_forge.tissue_forge import _vertex_solver_Mesh as Mesh
from tissue_forge.tissue_forge import _vertex_solver_MeshSolver as MeshSolver
from tissue_forge.tissue_forge import _vertex_solver_Structure as Structure
from tissue_forge.tissue_forge import _vertex_solver_StructureType as StructureType
from tissue_forge.tissue_forge import _vertex_solver_Surface as Surface
from tissue_forge.tissue_forge import _vertex_solver_SurfaceType as SurfaceType
from tissue_forge.tissue_forge import _vertex_solver_Vertex as Vertex
from tissue_forge.tissue_forge import _vertex_solver_Logger as Logger
from tissue_forge.tissue_forge import _vertex_solver_Quality as Quality
from tissue_forge.tissue_forge import _vertex_solver_BodyForce as BodyForce
from tissue_forge.tissue_forge import _vertex_solver_NormalStress as NormalStress
from tissue_forge.tissue_forge import _vertex_solver_SurfaceAreaConstraint as SurfaceAreaConstraint
from tissue_forge.tissue_forge import _vertex_solver_SurfaceTraction as SurfaceTraction
from tissue_forge.tissue_forge import _vertex_solver_VolumeConstraint as VolumeConstraint
from tissue_forge.tissue_forge import _vertex_solver_EdgeTension as EdgeTension
from tissue_forge.tissue_forge import _vertex_solver_Adhesion as Adhesion
from tissue_forge.tissue_forge import _vertex_solver_edgeStrain as edge_strain
from tissue_forge.tissue_forge import _vertex_solver_vertexStrain as vertex_strain

from tissue_forge.tissue_forge import _vertex_solver__MeshParticleType_get as MeshParticleType_get
from tissue_forge.tissue_forge import _vertex_solver__createQuadMesh_expMesh as _vertex_solver_create_quad_mesh_expmesh
from tissue_forge.tissue_forge import _vertex_solver__createQuadMesh_defMesh as _vertex_solver_create_quad_mesh_defmesh
from tissue_forge.tissue_forge import _vertex_solver__createPLPDMesh_expMesh as _vertex_solver_create_plpd_mesh_expmesh
from tissue_forge.tissue_forge import _vertex_solver__createPLPDMesh_defMesh as _vertex_solver_create_plpd_mesh_defmesh
from tissue_forge.tissue_forge import _vertex_solver__createHex2DMesh_expMesh as _vertex_solver_create_hex2d_mesh_expmesh
from tissue_forge.tissue_forge import _vertex_solver__createHex2DMesh_defMesh as _vertex_solver_create_hex2d_mesh_defmesh
from tissue_forge.tissue_forge import _vertex_solver__createHex3DMesh_expMesh as _vertex_solver_create_hex3d_mesh_expmesh
from tissue_forge.tissue_forge import _vertex_solver__createHex3DMesh_defMesh as _vertex_solver_create_hex3d_mesh_defmesh

init = MeshSolver.init

def create_quad_mesh(*args, **kwargs):
    mesh = kwargs.get("mesh")
    if mesh is None and len(args) > 0:
        if isinstance(args[0], Mesh):
            mesh = args[0]

    if mesh is not None:
        return _vertex_solver_create_quad_mesh_expmesh(*args, **kwargs)
    else:
        return _vertex_solver_create_quad_mesh_defmesh(*args, **kwargs)

def create_plpd_mesh(*args, **kwargs):
    mesh = kwargs.get("mesh")
    if mesh is None and len(args) > 0:
        if isinstance(args[0], Mesh):
            mesh = args[0]

    if mesh is not None:
        return _vertex_solver_create_plpd_mesh_expmesh(*args, **kwargs)
    else:
        return _vertex_solver_create_plpd_mesh_defmesh(*args, **kwargs)

def create_hex2d_mesh(*args, **kwargs):
    mesh = kwargs.get("mesh")
    if mesh is None and len(args) > 0:
        if isinstance(args[0], Mesh):
            mesh = args[0]

    if mesh is not None:
        return _vertex_solver_create_hex2d_mesh_expmesh(*args, **kwargs)
    else:
        return _vertex_solver_create_hex2d_mesh_defmesh(*args, **kwargs)

def create_hex3d_mesh(*args, **kwargs):
    mesh = kwargs.get("mesh")
    if mesh is None and len(args) > 0:
        if isinstance(args[0], Mesh):
            mesh = args[0]

    if mesh is not None:
        return _vertex_solver_create_hex3d_mesh_expmesh(*args, **kwargs)
    else:
        return _vertex_solver_create_hex3d_mesh_defmesh(*args, **kwargs)

