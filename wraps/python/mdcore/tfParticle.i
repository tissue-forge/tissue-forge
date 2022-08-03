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

#include "tfParticle.h"

#include <tfEngine.h>

%}


// Currently swig isn't playing nicely with FVector3 pointers for ParticleType::operator(), 
//  so we'll handle them manually for now
%rename(_call) TissueForge::ParticleType::operator();
%rename(_factory) TissueForge::ParticleType::factory(unsigned int, std::vector<FVector3>*, std::vector<FVector3>*, std::vector<int>*);
%rename(to_cluster) TissueForge::ParticleHandle::operator ClusterParticleHandle*();
%rename(to_cluster) TissueForge::ParticleType::operator ClusterParticleType*();

%ignore TissueForge::Particle_Colors;
%ignore TissueForge::Particle_Verify;
%ignore TissueForge::ParticleType_ForEngine;
%ignore TissueForge::ParticleType_New;
%ignore TissueForge::Particles_New;
%ignore TissueForge::_Particle_init;

%include "tfParticle.h"

%template(vectorParticle) std::vector<TissueForge::Particle>;
%template(vectorParticleHandle) std::vector<TissueForge::ParticleHandle>;
%template(vectorParticleType) std::vector<TissueForge::ParticleType>;

%extend TissueForge::Particle {
    %pythoncode %{
        def __reduce__(self):
            return Particle.fromString, (self.toString(),)
    %}
}

%extend TissueForge::ParticleHandle {
    %pythoncode %{
        @property
        def charge(self):
            """Particle charge"""
            return self.getCharge()

        @charge.setter
        def charge(self, charge):
            self.setCharge(charge)

        @property
        def mass(self):
            """Particle mass"""
            return self.getMass()

        @mass.setter
        def mass(self, mass):
            self.setMass(mass)

        @property
        def frozen(self):
            """Particle frozen flag"""
            return self.getFrozen()

        @frozen.setter
        def frozen(self, frozen):
            self.setFrozen(frozen)

        @property
        def frozen_x(self):
            """Particle frozen flag along x"""
            return self.getFrozenX()

        @frozen_x.setter
        def frozen_x(self, frozen):
            self.setFrozenX(frozen)

        @property
        def frozen_y(self):
            """Particle frozen flag along y"""
            return self.getFrozenY()

        @frozen_y.setter
        def frozen_y(self, frozen):
            self.setFrozenY(frozen)

        @property
        def frozen_z(self):
            """Particle frozen flag along z"""
            return self.getFrozenZ()

        @frozen_z.setter
        def frozen_z(self, frozen):
            self.setFrozenZ(frozen)

        @property
        def style(self):
            """Particle style"""
            return self.getStyle()

        @style.setter
        def style(self, style):
            if style.thisown:
                style.thisown = False
            self.setStyle(style)

        @property
        def age(self):
            """Particle age"""
            return self.getAge()

        @property
        def radius(self):
            """Particle radius"""
            return self.getRadius()

        @radius.setter
        def radius(self, radius):
            self.setRadius(radius)

        @property
        def name(self):
            """Particle name"""
            return self.getName()

        @property
        def name2(self):
            return self.getName2()

        @property
        def position(self):
            """Particle position"""
            return self.getPosition()

        @position.setter
        def position(self, position):
            self.setPosition(FVector3(position))

        @property
        def velocity(self):
            """Particle velocity"""
            return self.getVelocity()

        @velocity.setter
        def velocity(self, velocity):
            self.setVelocity(FVector3(velocity))

        @property
        def force(self):
            """Net force acting on particle"""
            return self.getForce()

        @property
        def force_init(self):
            """Persistent force acting on particle"""
            return self.getForceInit()

        @force_init.setter
        def force_init(self, force):
            self.setForceInit(FVector3(force))

        @property
        def id(self):
            """Particle id"""
            return self.getId()

        @property
        def type_id(self):
            """Particle type id"""
            return self.getTypeId()

        @property
        def cluster_id(self):
            """Cluster particle id, if any; -1 if particle is not in a cluster"""
            return self.getClusterId()

        @property
        def flags(self):
            return self.getFlags()

        @property
        def species(self):
            """Particle species"""
            return self.getSpecies()

        @property
        def bonds(self):
            """Bonds attached to particle"""
            return self.getBonds()

        @property
        def angles(self):
            """Angles attached to particle"""
            return self.getAngles()

        @property
        def dihedrals(self):
            """Dihedrals attached to particle"""
            return self.getDihedrals()
    %}
}

%extend TissueForge::ParticleType {
    %pythoncode %{
        def __call__(self, *args, **kwargs):
            position = kwargs.get('position')
            velocity = kwargs.get('velocity')
            part_str = kwargs.get('part_str')
            cluster_id = kwargs.get('cluster_id')
            
            n_args = len(args)
            if n_args > 0:
                if isinstance(args[0], str):
                    part_str = args[0]
                    if n_args > 1:
                        if isinstance(args[1], int):
                            cluster_id = args[1]
                        else:
                            raise TypeError
                elif isinstance(args[0], int):
                    cluster_id = args[0]
                else:
                    position = args[0]
                    if n_args > 1:
                        if isinstance(args[1], int):
                            cluster_id = args[1]
                        else:
                            velocity = args[1]
                            if n_args > 2:
                                cluster_id = args[2]

            pos, vel = None, None
            if position is not None:
                pos = FVector3(list(position)) if not isinstance(position, FVector3) else position

            if velocity is not None:
                vel = FVector3(list(velocity)) if not isinstance(velocity, FVector3) else velocity

            if part_str is not None:
                return self._call(part_str, cluster_id)
            return self._call(pos, vel, cluster_id)

        def factory(self, nr_parts=0, positions=None, velocities=None, cluster_ids=None):
            _positions = None
            if positions is not None:
                _positions = vectorFVector3()
                [_positions.push_back(FVector3(x)) for x in positions]

            _velocities = None
            if velocities is not None:
                _velocities = vectorFVector3()
                [_velocities.push_back(FVector3(x)) for x in velocities]

            _cluster_ids = None
            if cluster_ids is not None:
                _cluster_ids = vectori()
                [_cluster_ids.push_back(x) for x in cluster_ids]

            return self._factory(nr_parts=nr_parts, positions=_positions, velocities=_velocities, clusterIds=_cluster_ids)

        @property
        def frozen(self):
            """Particle type frozen flag"""
            return self.getFrozen()

        @frozen.setter
        def frozen(self, frozen):
            self.setFrozen(frozen)

        @property
        def frozen_x(self):
            """Particle type frozen flag along x"""
            return self.getFrozenX()

        @frozen_x.setter
        def frozen_x(self, frozen):
            self.setFrozenX(frozen)

        @property
        def frozen_y(self):
            """Particle type frozen flag along y"""
            return self.getFrozenY()

        @frozen_y.setter
        def frozen_y(self, frozen):
            self.setFrozenY(frozen)

        @property
        def frozen_z(self):
            """Particle type frozen flag along z"""
            return self.getFrozenZ()

        @frozen_z.setter
        def frozen_z(self, frozen):
            self.setFrozenZ(frozen)

        @property
        def temperature(self):
            """Particle type temperature"""
            return self.getTemperature()

        @property
        def target_temperature(self):
            """Particle type target temperature"""
            return self.getTargetTemperature()

        @target_temperature.setter
        def target_temperature(self, temperature):
            self.setTargetTemperature(temperature)

        def __reduce__(self):
            return ParticleType.fromString, (self.toString(),)
    %}
}

%pythoncode %{
    Newtonian = PARTICLE_NEWTONIAN
    Overdamped = PARTICLE_OVERDAMPED
%}

// In python, we'll specify particle types using class attributes of the same name 
//  as underlying C++ struct. ParticleType is the helper class through which this 
//  functionality occurs. ParticleType is defined in particle_type.py

