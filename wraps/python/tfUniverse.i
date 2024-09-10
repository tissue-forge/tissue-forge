/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
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

#include "tfUniverse.h"

%}


%rename(_tfUniverse) TissueForge::Universe;

%ignore TissueForge::UniverseConfig;
%ignore TissueForge::_Universe;
%ignore TissueForge::getUniverse;

%include "tfUniverse.h"

%pythoncode %{
    class UniverseInterface:

        @property
        def temperature(self):
            """
            Universe temperature
            """
            return _tfUniverse.getTemperature()

        @temperature.setter
        def temperature(self, _val: float):
            _tfUniverse.setTemperature(_val)

        @property
        def boltzmann(self):
            """
            Boltzmann constant
            """
            return _tfUniverse.getBoltzmann()

        @boltzmann.setter
        def boltzmann(self, _val: float):
            _tfUniverse.setBoltzmann(_val)

        @property
        def time(self):
            """
            Current time
            """
            return _tfUniverse.getTime()

        @property
        def dt(self):
            """
            Time step
            """
            return _tfUniverse.getDt()

        @property
        def boundary_conditions(self):
            """
            Boundary conditions
            """
            return _tfUniverse.getBoundaryConditions()

        @property
        def kinetic_energy(self):
            """
            Universe kinetic energy
            """
            return _tfUniverse.getKineticEnergy()

        @property
        def center(self) -> fVector3:
            """
            Universe center point
            """
            return _tfUniverse.getCenter()

        @property
        def num_types(self) -> int:
            """
            Number of particle types
            """
            return _tfUniverse.getNumTypes()

        @property
        def cutoff(self) -> float:
            """
            Global interaction cutoff distance
            """
            return _tfUniverse.getCutoff()

        @property
        def flux_steps(self) -> int:
            """
            Number of flux steps per simulation step
            """
            return _tfUniverse.getNumFluxSteps()

        @property
        def dim(self) -> fVector3:
            """
            Universe dimensions
            """
            return _tfUniverse.dim()

        @property
        def volume(self) -> float:
            """
            Universe volume
            """
            return _tfUniverse.volume()

        @property
        def name(self) -> str:
            """
            name of the model / script
            """
            return _tfUniverse.getName()

        def virial(self, origin=None, radius: float = None, types=None) -> fMatrix3:
            """
            Computes the virial tensor for the either the entire simulation 
            domain, or a specific local virial tensor at a location and 
            radius. Optionally can accept a list of particle types to restrict the 
            virial calculation for specify types.
            
            :param origin: An optional length-3 array for the origin. Defaults to the center of the simulation domain if not given.
            :param radius: An optional number specifying the size of the region to compute the virial tensor for. Defaults to the entire simulation domain.
            :param types: An optional list of :class:`Particle` types to include in the calculation. Defaults to every particle type.
            :return: virial tensor
            """
            _origin = FVector3(origin) if origin else origin
            if types and not isinstance(types, vectorParticleType):
                _types = vectorParticleType()
                [_types.push_back(pt) for pt in types]
            else:
                _types = types
            return _tfUniverse.virial(_origin, radius, _types)

        def step(self, until: float = 0):
            """
            Performs a single time step of the universe if no arguments are 
            given. Optionally runs until ``until``.
            
            :param until: period to execute, in units of simulation time (default executes one time step).
            """
            return _tfUniverse.step(until, 0)

        def stop(self):
            """
            Stops the universe time evolution. This essentially freezes the universe, 
            everything remains the same, except time no longer moves forward.
            """
            return _tfUniverse.stop()

        def start(self):
            """
            Starts the universe time evolution, and advanced the universe forward by 
            timesteps in ``dt``. All methods to build and manipulate universe objects 
            are valid whether the universe time evolution is running or stopped.
            """
            return _tfUniverse.start()

        @property
        def particles(self):
            """
            Gets all particles in the universe
            
            :rtype: ParticleList
            """
            return ParticleList(_tfUniverse.particleIds())

        def reset_species(self):
            """
            Reset all species in all particles
            """
            return _tfUniverse.resetSpecies()

        def grid(self, shape):
            """
            Gets a three-dimesional array of particle lists, of all the particles in the system. 
            
            :param shape: shape of grid
            :return: three-dimension array of particle lists according to discretization and location
            """
            return _tfUniverse.grid(iVector3(shape))

        @property
        def bonds(self):
            """
            All bonds in the universe
            """
            return _tfUniverse.bonds()

        @property
        def angles(self):
            """
            All angles in the universe
            """
            return _tfUniverse.angles()

        @property
        def dihedrals(self):
            """
            All dihedrals in the universe
            """
            return _tfUniverse.dihedrals()


    Universe = UniverseInterface()

%}
