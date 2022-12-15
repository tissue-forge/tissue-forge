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
    from tissue_forge.tissue_forge import _vertex_solver_Body
    if isinstance(args[1], _vertex_solver_Body):
        return _bind_body_inst(*args)
    return _bind_body_type(*args)


def surface(*args):
    from tissue_forge.tissue_forge import _vertex_solver_Surface
    if isinstance(args[1], _vertex_solver_Surface):
        return _bind_surface_inst(*args)
    return _bind_surface_type(*args)
