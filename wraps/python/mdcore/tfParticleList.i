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

#include "tfParticleList.h"

%}


%template(vectorParticleList) std::vector<TissueForge::ParticleList>;
%template(vector2ParticleList) std::vector<std::vector<TissueForge::ParticleList>>;
%template(vector3ParticleList) std::vector<std::vector<std::vector<TissueForge::ParticleList>>>;
%template(vectorParticleList_p) std::vector<TissueForge::ParticleList*>;
%template(vector2ParticleList_p) std::vector<std::vector<TissueForge::ParticleList*>>;
%template(vector3ParticleList_p) std::vector<std::vector<std::vector<TissueForge::ParticleList*>>>;

%include "tfParticleList.h"

%extend TissueForge::ParticleList {
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
            """Virial tensor of particles in list"""
            return self.getVirial()

        @property
        def radius_of_gyration(self):
            """Radius of gyration of particles in list"""
            return self.getRadiusOfGyration()

        @property
        def center_of_mass(self):
            """Center of mass of particles in list"""
            return self.getCenterOfMass()

        @property
        def centroid(self):
            """Centroid of particles in list"""
            return self.getCentroid()

        @property
        def moment_of_inertia(self):
            """Moment of inertia of particles in list"""
            return self.getMomentOfInertia()

        @property
        def positions(self):
            """Position of each particle in list"""
            return self.getPositions()

        @property
        def velocities(self):
            """Velocity of each particle in list"""
            return self.getVelocities()

        @property
        def forces(self):
            """Net forces acting on each particle in list"""
            return self.getForces()

        def __reduce__(self):
            return ParticleList.fromString, (self.toString(),)
    %}
}
