/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

#include "tf_metrics.h"

%}

%rename(_metrics_relative_position) TissueForge::metrics::relativePosition;
%rename(_metrics_neighborhood_particles) TissueForge::metrics::neighborhoodParticles;
%rename(_metrics_calculate_virial) TissueForge::metrics::calculateVirial;
%rename(_metrics_particles_virial) TissueForge::metrics::particlesVirial;
%rename(_metrics_particles_radius_of_gyration) TissueForge::metrics::particlesRadiusOfGyration;
%rename(_metrics_particles_center_of_mass) TissueForge::metrics::particlesCenterOfMass;
%rename(_metrics_particles_center_of_geometry) TissueForge::metrics::particlesCenterOfGeometry;
%rename(_metrics_particles_moment_of_inertia) TissueForge::metrics::particlesMomentOfInertia;
%rename(_metrics_cartesian_to_spherical) TissueForge::metrics::cartesianToSpherical;
%rename(_metrics_particle_neighbors) TissueForge::metrics::particleNeighbors;
%rename(_metrics_eigenvals) TissueForge::metrics::eigenVals;
%rename(_metrics_eigenvecs_vals) TissueForge::metrics::eigenVecsVals;

%include "tf_metrics.h"
