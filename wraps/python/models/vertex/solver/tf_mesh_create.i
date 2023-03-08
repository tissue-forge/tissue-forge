/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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

#include <models/vertex/solver/tf_mesh_create.h>
%}

%rename(_vertex_solver__createQuadMesh) TissueForge::models::vertex::createQuadMesh(SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const FloatP_t&, const FloatP_t&, const char*ax_1, const char*ax_2);
%rename(_vertex_solver__createPLPDMesh) TissueForge::models::vertex::createPLPDMesh(BodyType*, SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const unsigned int&, const FloatP_t&, const FloatP_t&, const FloatP_t&, const char*, const char*);
%rename(_vertex_solver__createHex2DMesh) TissueForge::models::vertex::createHex2DMesh(SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const FloatP_t&, const char*, const char*);
%rename(_vertex_solver__createHex3DMesh) TissueForge::models::vertex::createHex3DMesh(BodyType*, SurfaceType*, const FVector3&, const unsigned int&, const unsigned int&, const unsigned int&, const FloatP_t&, const FloatP_t&, const char*, const char*);

%include <models/vertex/solver/tf_mesh_create.h>
