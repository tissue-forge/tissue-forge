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

%define vertex_solver_MeshObjActor_prep(baseName) 

%inline %{

static std::vector<TissueForge::models::vertex:: ## baseName ## *> _vertex_solver_MeshObjActor_get ## baseName(TissueForge::models::vertex::Surface *obj) {
    return TissueForge::models::vertex::MeshObjActor::get<TissueForge::models::vertex:: ## baseName ##>(obj);
}

static std::vector<TissueForge::models::vertex:: ## baseName ## *> _vertex_solver_MeshObjActor_get ## baseName(TissueForge::models::vertex::Body *obj) {
    return TissueForge::models::vertex::MeshObjActor::get<TissueForge::models::vertex:: ## baseName ##>(obj);
}

static std::vector<TissueForge::models::vertex:: ## baseName ## *> _vertex_solver_MeshObjActor_get ## baseName(TissueForge::models::vertex::SurfaceType *objType) {
    return TissueForge::models::vertex::MeshObjActor::get<TissueForge::models::vertex:: ## baseName ##>(objType);
}

static std::vector<TissueForge::models::vertex:: ## baseName ## *> _vertex_solver_MeshObjActor_get ## baseName(TissueForge::models::vertex::BodyType *objType) {
    return TissueForge::models::vertex::MeshObjActor::get<TissueForge::models::vertex:: ## baseName ##>(objType);
}

%}

%template(vectorMesh ## baseName) std::vector<TissueForge::models::vertex:: ## baseName ## *>;

%enddef


%include "tfAdhesion.i"
%include "tfBodyForce.i"
%include "tfConvexPolygonConstraint.i"
%include "tfFlatSurfaceConstraint.i"
%include "tfNormalStress.i"
%include "tfPerimeterConstraint.i"
%include "tfSurfaceAreaConstraint.i"
%include "tfSurfaceTraction.i"
%include "tfVolumeConstraint.i"
%include "tfEdgeTension.i"
