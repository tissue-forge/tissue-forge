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

#include "tfNormalStress.h"

#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfBody.h>

#include <tfEngine.h>


using namespace TissueForge::models::vertex;


HRESULT NormalStress::energy(const MeshObj *source, const MeshObj *target, FloatP_t &e) {
    FVector3 fv;
    force(source, target, fv.data());
    e = fv.dot(((Vertex*)target)->particle()->getVelocity()) * _Engine.dt;
    return S_OK;
}

HRESULT NormalStress::force(const MeshObj *source, const MeshObj *target, FloatP_t *f) {
    Surface *s = (Surface*)source;
    Vertex *v = (Vertex*)target;
    
    auto bodies = s->getBodies();
    if(bodies.size() == 2) 
        return S_OK;
    
    FVector3 snormal = s->getNormal();
    if(snormal.isZero()) 
        return S_OK;

    snormal = snormal.normalized();
    if(bodies.size() == 1) 
        snormal *= s->volumeSense(bodies[0]);

    FVector3 vForce = snormal * mag * s->getVertexArea(v);
    f[0] += vForce[0];
    f[1] += vForce[1];
    f[2] += vForce[2];
    return S_OK;
}
