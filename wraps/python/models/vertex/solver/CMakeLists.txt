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

install(
    FILES 
    __init__.py 
    bind.py 
    mesh_types.py 
    DESTINATION 
    ${TF_SITEPACKAGES}/tissue_forge/models/vertex/solver
)
install(FILES 
    examples/CapillaryLoop.obj 
    examples/capillary_loop.py 
    examples/cell_migration.py 
    examples/cell_sorting.py 
    examples/cell_splitting.py 
    DESTINATION 
    ${TF_SITEPACKAGES}/tissue_forge/examples/models/vertex/solver
)
