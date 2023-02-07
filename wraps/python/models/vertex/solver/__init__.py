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

from tissue_forge.tissue_forge import _vertex_solver_Body
from tissue_forge.tissue_forge import _vertex_solver_BodyHandle
from tissue_forge.tissue_forge import _vertex_solver_BodyType
from tissue_forge.tissue_forge import _vertex_solver_Mesh
from tissue_forge.tissue_forge import _vertex_solver_MeshObjActor
from tissue_forge.tissue_forge import _vertex_solver_MeshObjType
from tissue_forge.tissue_forge import _vertex_solver_MeshSolver
from tissue_forge.tissue_forge import _vertex_solver_MeshSolverTimers
from tissue_forge.tissue_forge import _vertex_solver_Surface
from tissue_forge.tissue_forge import _vertex_solver_SurfaceHandle
from tissue_forge.tissue_forge import _vertex_solver_SurfaceType
from tissue_forge.tissue_forge import _vertex_solver_Vertex
from tissue_forge.tissue_forge import _vertex_solver_VertexHandle
from tissue_forge.tissue_forge import _vertex_solver_Logger
from tissue_forge.tissue_forge import _vertex_solver_Quality
from tissue_forge.tissue_forge import _vertex_solver_BodyForce
from tissue_forge.tissue_forge import _vertex_solver_NormalStress
from tissue_forge.tissue_forge import _vertex_solver_PerimeterConstraint
from tissue_forge.tissue_forge import _vertex_solver_SurfaceAreaConstraint
from tissue_forge.tissue_forge import _vertex_solver_SurfaceTraction
from tissue_forge.tissue_forge import _vertex_solver_VolumeConstraint
from tissue_forge.tissue_forge import _vertex_solver_EdgeTension
from tissue_forge.tissue_forge import _vertex_solver_Adhesion
from tissue_forge.tissue_forge import _vertex_solver_edgeStrain as _vertex_solver_edgeStrain
from tissue_forge.tissue_forge import _vertex_solver_vertexStrain as _vertex_solver_vertexStrain

from tissue_forge.tissue_forge import _vertex_solver__MeshParticleType_get as MeshParticleType_get
from tissue_forge.tissue_forge import _vertex_solver__createQuadMesh as create_quad_mesh
from tissue_forge.tissue_forge import _vertex_solver__createPLPDMesh as create_plpd_mesh
from tissue_forge.tissue_forge import _vertex_solver__createHex2DMesh as create_hex2d_mesh
from tissue_forge.tissue_forge import _vertex_solver__createHex3DMesh as create_hex3d_mesh

__all__ = ['bind']

class MeshObjActor(_vertex_solver_MeshObjActor):
    """
    Base definition of how a mesh object acts on another mesh object
    """
    pass

class MeshObjType(_vertex_solver_MeshObjType):
    """
    Base mesh object type definition. 
    
    The type definition of a mesh object should derive from this class
    """
    pass

class Body(_vertex_solver_Body):
    """
    The mesh body is a volume-enclosing object of mesh surfaces. 
    
    The mesh body consists of at least four mesh surfaces. 
    
    The mesh body can have a state vector, which represents a uniform amount of substance 
    enclosed in the volume of the body. 
    
    """
    pass

class BodyHandle(_vertex_solver_BodyHandle):
    """
    A handle to a :class:`Body`. 
    
    The engine allocates :class:`Body` memory in blocks, and :class:`Body`
    values get moved around all the time, so their addresses change.
    
    This is a safe way to work with a :class:`Body`.
    """
    pass

class BodyType(_vertex_solver_BodyType):
    """
    Mesh body type
    
    Can be used as a factory to create mesh body instances with 
    processes and properties that correspond to the type. 
    """
    pass

class Mesh(_vertex_solver_Mesh):
    """
    Contains all :class:`Vertex`, :class:`Surface` and :class:`Body` instances
    """
    pass

class MeshSolver(_vertex_solver_MeshSolver):
    """
    Vertex model mesh solver
    
    A singleton solver performs all vertex model dynamics simulation at runtime. 
    """
    pass

class MeshSolverTimers(_vertex_solver_MeshSolverTimers):
    """
    Mesh solver performance timers
    """
    pass

class Surface(_vertex_solver_Surface):
    """
    The mesh surface is an area-enclosed object of implicit mesh edges defined by mesh vertices. 
    
    The mesh surface consists of at least three mesh vertices. 
    
    The mesh surface is always flat. 
    
    The mesh surface can have a state vector, which represents a uniform amount of substance 
    attached to the surface. 
    
    """
    pass

class SurfaceHandle(_vertex_solver_SurfaceHandle):
    """
    A handle to a :class:`Surface`. 
    
    The engine allocates :class:`Surface` memory in blocks, and :class:`Surface`
    values get moved around all the time, so their addresses change.
    
    This is a safe way to work with a :class:`Surface`.
    """
    pass

class SurfaceType(_vertex_solver_SurfaceType):
    """
    Mesh surface type. 
    
    Can be used as a factory to create mesh surface instances with 
    processes and properties that correspond to the type. 
    """
    pass

class Vertex(_vertex_solver_Vertex):
    """
    The mesh vertex is a volume of a mesh centered at a point in a space.
    """
    pass

class VertexHandle(_vertex_solver_VertexHandle):
    """
    A handle to a :class:`Vertex`. 
    
    The engine allocates :class:`Vertex` memory in blocks, and :class:`Vertex`
    values get moved around all the time, so their addresses change.
    
    This is a safe way to work with a :class:`Vertex`.
    """
    pass

class Logger(_vertex_solver_Logger):
    """
    The Tissue Forge vertex model solver logger. 
    """
    pass

class Quality(_vertex_solver_Quality):
    """
    An object that schedules topological operations on a mesh to maintain its quality
    """
    pass

class BodyForce(_vertex_solver_BodyForce):
    """
    Imposes a body force on :class:`Body` instances
    """
    pass

class NormalStress(_vertex_solver_NormalStress):
    """
    Models a stress acting on a :class:`Surface` along its normal
    """
    pass

class PerimeterConstraint(_vertex_solver_PerimeterConstraint):
    """
    Imposes a perimeter constraint on 'Surface' instances.

    The perimeter constraint is implemented for two-dimensional objects
    as minimization of the Hamiltonian,

    .. math::

        \lambda \left( L - L_o \right)^2

    Here :math:`\lambda` is a parameter,
    :math:`L` is the perimeter of an object and
    :math:`L_o` is a target perimeter.
    """
    pass

class SurfaceAreaConstraint(_vertex_solver_SurfaceAreaConstraint):
    """
    Imposes a surface area constraint on 'Body' or 'Surface' instances.

    The surface area constraint is implemented for two- and three-dimensional objects
    as minimization of the Hamiltonian,

    .. math::

        \lambda \left( A - A_o \right)^2

    Here :math:`\lambda` is a parameter,
    :math:`A` is the area an object and
    :math:`A_o` is a target area.
    """
    pass

class SurfaceTraction(_vertex_solver_SurfaceTraction):
    """
    Models a traction force
    """
    pass

class VolumeConstraint(_vertex_solver_VolumeConstraint):
    """
    Imposes a volume constraint.

    The volume constraint is implemented for three-dimensional objects
    as minimization of the Hamiltonian,

    .. math::

        \lambda \left( V - V_o \right)^2

    Here :math:`\lambda` is a parameter,
    :math:`V` is the volume an object and
    :math:`V_o` is a target volume.
    """
    pass

class EdgeTension(_vertex_solver_EdgeTension):
    """
    Models tension between connected vertices.

    Edge tension is implemented for two-dimensional objects as minimization of the Hamiltonian,

    .. math::

        \lambda L^n

    Here :math:`\lambda` is a parameter,
    :math:`L` is the length of an edge shared by two objects and
    :math:`n > 0` is the order of the model.
    """
    pass

class Adhesion(_vertex_solver_Adhesion):
    """
    Models adhesion between pairs of 'Surface' or 'Body' instances by type.

    Adhesion is implemented for two-dimensional objects as minimization of the Hamiltonian,

    .. math::

        \lambda L

    Here :math:`\lambda` is a parameter and
    :math:`L` is the length of edges shared by two objects.

    Adhesion is implemented for three-dimensional objects as minimization of the Hamiltonian,

    .. math::

        \lambda A

    Here :math:`A` is the area shared by two objects.
    """
    pass


init = MeshSolver.init
