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

#include <tf_bind.h>
#include <langs/py/tf_bindPy.h>

%}


%rename(_bind_particles) TissueForge::py::particles;
%rename(_bind_types) TissueForge::py::types;
%rename(_bind_boundary_conditions) TissueForge::py::boundary_conditions;
%rename(_bind_boundary_condition) TissueForge::py::boundary_condition;
%rename(_bind_force) TissueForge::py::force;

#include <tfBond.h>
#include <tfParticle.h>

%include "langs/py/tf_bindPy.h"

%template(pairParticleType_ParticleType) std::pair<TissueForge::ParticleType*, TissueForge::ParticleType*>;
%template(vectorPairParticleType_ParticleType) std::vector<std::pair<TissueForge::ParticleType*, TissueForge::ParticleType*>*>;


%pythoncode %{
    def _bind_bonds(potential, particles, cutoff, pairs=None, half_life=None, bond_energy=None, flags=0):
        if not isinstance(particles, ParticleList):
            particle_list = ParticleList()
            [particle_list.insert(p) for p in particles]
        else:
            particle_list = particles

        ppairs = None
        if pairs is not None:
            ppairs = vectorPairParticleType_ParticleType([pairParticleType_ParticleType(p) for p in pairs])
        return _bondsPy(potential, particle_list, cutoff, ppairs, half_life, bond_energy, flags)

    def _bind_sphere(potential, n, center=None, radius=None, phi=None, type=None):
        if phi is not None:
            phi0, phi1 = phi[0], phi[1]
        else:
            phi0, phi1 = None, None
        particles = ParticleList()
        bonds = _spherePy(particles, potential, n, center, radius, phi0, phi1, type)
        particles.thisown = 0
        return particles, bonds
%}
