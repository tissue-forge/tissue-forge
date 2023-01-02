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

from tissue_forge.tissue_forge import _vertex_solver_bind_body_type as _bind_body_type
from tissue_forge.tissue_forge import _vertex_solver_bind_body_inst as _bind_body_inst
from tissue_forge.tissue_forge import _vertex_solver_bind_surface_type as _bind_surface_type
from tissue_forge.tissue_forge import _vertex_solver_bind_surface_inst as _bind_surface_inst
from tissue_forge.tissue_forge import _vertex_solver_bind_types as types


def body(*args):
    """
    Bind an actor to either a body or body type

    :param args: an actor and either a body or body type
    """
    from tissue_forge.tissue_forge import _vertex_solver_Body, _vertex_solver_BodyHandle

    actor = args[0]
    _target = args[1]
    func = None
    if isinstance(_target, _vertex_solver_Body):
        target = _vertex_solver_BodyHandle(_target.id)
        func = _bind_body_inst
    elif isinstance(_target, _vertex_solver_BodyHandle):
        target = _target
        func = _bind_body_inst
    else:
        target = _target
        func = _bind_body_type
    
    return func(actor, target)


def surface(*args):
    """
    Bind an actor to either a surface or surface type

    :param args: an actor and either a body or body type
    """
    from tissue_forge.tissue_forge import _vertex_solver_Surface, _vertex_solver_SurfaceHandle
    
    actor = args[0]
    _target = args[1]
    func = None
    if isinstance(_target, _vertex_solver_Surface):
        target = _vertex_solver_SurfaceHandle(_target.id)
        func = _bind_surface_inst
    elif isinstance(_target, _vertex_solver_SurfaceHandle):
        target = _target
        func = _bind_surface_inst
    else:
        target = _target
        func = _bind_surface_type

    return func(actor, target)
