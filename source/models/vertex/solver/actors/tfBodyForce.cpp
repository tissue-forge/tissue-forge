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

#include "tfBodyForce.h"

#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfBody.h>

#include <tfEngine.h>


using namespace TissueForge::models::vertex;


HRESULT BodyForce::energy(MeshObj *source, MeshObj *target, float &e) {
    FVector3 fv;
    force(source, target, fv.data());
    e = fv.dot(((Vertex*)target)->particle()->getVelocity()) * _Engine.dt;
    return S_OK;
}

HRESULT BodyForce::force(MeshObj *source, MeshObj *target, float *f) {
    Body *b = (Body*)source;
    float bArea = b->getArea();
    if(bArea == 0.f) {
        return S_OK;
    }
    
    FVector3 fv = comps * b->getVertexArea((Vertex*)target) / bArea;

    f[0] += fv[0];
    f[1] += fv[1];
    f[2] += fv[2];
    return S_OK;
}
