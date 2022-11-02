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

#include "tfFlatSurfaceConstraint.h"

#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfVertex.h>

#include <tfEngine.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


HRESULT FlatSurfaceConstraint::energy(const MeshObj *source, const MeshObj *target, FloatP_t &e) {
    Surface *s = (Surface*)source;
    Vertex *v = (Vertex*)target;

    FloatP_t _e = (s->getCentroid() - v->getPosition()).dot(s->getNormal());
    
    e += v->particle()->getMass() / 2.f / _Engine.dt * lam * _e * _e;
    
    return S_OK;
}

HRESULT FlatSurfaceConstraint::force(const MeshObj *source, const MeshObj *target, FloatP_t *f) {
    Surface *s = (Surface*)source;
    Vertex *v = (Vertex*)target;

    FVector3 sn = s->getNormal();
    FVector3 force = (sn * ((s->getCentroid() - v->getPosition()).dot(sn))) * v->particle()->getMass() / _Engine.dt * lam;

    f[0] += force[0];
    f[1] += force[1];
    f[2] += force[2];
    
    return S_OK;
}
