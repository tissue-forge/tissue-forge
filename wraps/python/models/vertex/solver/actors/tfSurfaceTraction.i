/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego and Tien Comlekoglu
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

#include <models/vertex/solver/actors/tfSurfaceTraction.h>
%}

// todo: correct so that this isn't necessary
%ignore TissueForge::models::vertex::SurfaceTraction::energy(const MeshObj *, const MeshObj *, FloatP_t &);
%ignore TissueForge::models::vertex::SurfaceTraction::force(const MeshObj *, const MeshObj *, FloatP_t *);

%rename(_vertex_solver_SurfaceTraction) TissueForge::models::vertex::SurfaceTraction;

%include <models/vertex/solver/actors/tfSurfaceTraction.h>

vertex_solver_MeshObjActor_prep(SurfaceTraction)
