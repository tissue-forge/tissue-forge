/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

%{

#include <tfBoundaryConditions.h>
#include <langs/py/tfBoundaryConditionsPy.h>

%}


%ignore apply_boundary_particle_crossing;

%include <tfBoundaryConditions.h>
%include <langs/py/tfBoundaryConditionsPy.h>

%extend TissueForge::BoundaryCondition {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()

        @property
        def kind_str(self) -> str:
            """
            The name of the kind of boundary condition. 
            """
            return self.kindStr()
    %}
}


%extend TissueForge::BoundaryConditions {
    %pythoncode %{
        def __str__(self) -> str:
            return self.str()

        def __reduce__(self):
            return BoundaryConditions.fromString, (self.toString(),)
    %}
}

%pythoncode %{
    from enum import Enum as EnumPy

    class BoundaryTypeFlags(EnumPy):
    
        none = BoundaryTypeFlags_BOUNDARY_NONE
        periodic_x = BoundaryTypeFlags_PERIODIC_X
        periodic_y = BoundaryTypeFlags_PERIODIC_Y
        periodic_z = BoundaryTypeFlags_PERIODIC_Z
        periodic = BoundaryTypeFlags_PERIODIC_FULL
        ghost_x = BoundaryTypeFlags_PERIODIC_GHOST_X
        ghost_y = BoundaryTypeFlags_PERIODIC_GHOST_Y
        ghost_z = BoundaryTypeFlags_PERIODIC_GHOST_Z
        ghost = BoundaryTypeFlags_PERIODIC_GHOST_FULL
        freeslip_x = BoundaryTypeFlags_FREESLIP_X
        freeslip_y = BoundaryTypeFlags_FREESLIP_Y
        freeslip_z = BoundaryTypeFlags_FREESLIP_Z
        freeslip = BoundaryTypeFlags_FREESLIP_FULL
    
%}
