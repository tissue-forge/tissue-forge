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

#include "tfParticleTypeList.h"

%}


%include "tfParticleTypeList.h"

%extend TissueForge::ParticleTypeList {
    %pythoncode %{
        def __len__(self) -> int:
            return self.nr_parts

        def __getitem__(self, i: int):
            if i >= len(self):
                raise IndexError('Valid indices < ' + str(len(self)))
            return self.item(i)

        def __contains__(self, item):
            return self.has(item)

        @property
        def virial(self):
            """Virial tensor of particles corresponding to all types in list"""
            return self.getVirial()

        @property
        def radius_of_gyration(self):
            """Radius of gyration of particles corresponding to all types in list"""
            return self.getRadiusOfGyration()

        @property
        def center_of_mass(self):
            """Center of mass of particles corresponding to all types in list"""
            return self.getCenterOfMass()

        @property
        def centroid(self):
            """Centroid of particles corresponding to all types in list"""
            return self.getCentroid()

        @property
        def moment_of_inertia(self):
            """Moment of inertia of particles corresponding to all types in list"""
            return self.getMomentOfInertia()

        @property
        def positions(self):
            """Position of each particle corresponding to all types in list"""
            return self.getPositions()

        @property
        def velocities(self):
            """Velocity of each particle corresponding to all types in list"""
            return self.getVelocities()

        @property
        def forces(self):
            """Total net force acting on each particle corresponding to all types in list"""
            return self.getForces()

        def __reduce__(self):
            return ParticleTypeList.fromString, (self.toString(),)
    %}
}
