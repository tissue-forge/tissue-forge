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

#include "tfCluster.h"

%}


// Currently swig isn't playing nicely with FVector3 pointers for ::operator(), 
//  so we'll handle them manually for now
%rename(_call) TissueForge::ClusterParticleHandle::operator();
%rename(_call) TissueForge::ClusterParticleType::operator();

%ignore _Cluster_init;
%ignore Cluster_fromString;
%ignore ClusterParticleType_fromString;

%include "tfCluster.h"

%template(list_ParticleType) std::list<TissueForge::ParticleType*>;

%extend TissueForge::Cluster {
    %pythoncode %{
        def __reduce__(self):
            return Cluster.fromString, (self.toString(),)
    %}
}

%extend TissueForge::ClusterParticleHandle {
    %pythoncode %{
        def __call__(self, particle_type, *args, **kwargs):
            position = kwargs.get('position')
            velocity = kwargs.get('velocity')
            part_str = kwargs.get('part_str')
            
            n_args = len(args)
            if n_args > 0:
                if isinstance(args[0], str):
                    part_str = args[0]
                    if n_args > 1:
                        raise TypeError
                else:
                    position = args[0]
                    if n_args > 1:
                        velocity = args[1]
                        if n_args > 2:
                            raise TypeError

            pos, vel = None, None
            if position is not None:
                pos = FVector3(list(position))

            if velocity is not None:
                vel = FVector3(list(velocity))

            args = []
            if pos is not None:
                args.append(pos)
            if vel is not None:
                args.append(vel)
            if args:
                return self._call(particle_type, *args)

            if part_str is not None:
                args.append(part_str)

            return self._call(particle_type, *args)

        @property
        def radius_of_gyration(self):
            """Radius of gyration"""
            return self.getRadiusOfGyration()

        @property
        def center_of_mass(self):
            """Center of mass"""
            return self.getCenterOfMass()

        @property
        def centroid(self):
            """Centroid"""
            return self.getCentroid()

        @property
        def moment_of_inertia(self):
            """Moment of inertia"""
            return self.getMomentOfInertia()
    %}
}

%extend TissueForge::ClusterParticleType {
    %pythoncode %{
        def __call__(self, position=None, velocity=None, cluster_id=None):
            ph = ParticleType.__call__(self, position, velocity, cluster_id)
            return ClusterParticleHandle(ph.id, ph.type_id)

        def __reduce__(self):
            return ClusterParticleType.fromString, (self.toString(),)
    %}
}

// In python, we'll specify cluster types using class attributes of the same name 
//  as underlying C++ struct. ClusterType is the helper class through which this 
//  functionality occurs. ClusterType is defined in particle_type.py
