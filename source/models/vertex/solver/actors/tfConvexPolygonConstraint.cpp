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

#include "tfConvexPolygonConstraint.h"

#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfVertex.h>

#include <tfEngine.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


static inline bool ConvexPolygonConstraint_acts(Vertex *vc, Surface *s, FVector3 &rel_c2ab) {
    Vertex *va, *vb;
    std::tie(va, vb) = s->neighborVertices(vc);
    const FVector3 posvc = vc->getPosition();

    // Perpindicular vector from vertex to line connecting neighbors should point 
    //  in the same direction as the vector from the vertex to the centroid
    rel_c2ab = posvc.lineShortestDisplacementTo(va->getPosition(), vb->getPosition());
    return rel_c2ab.dot(s->getCentroid() - posvc) < 0;
}

HRESULT ConvexPolygonConstraint::energy(const MeshObj *source, const MeshObj *target, FloatP_t &e) {
    Vertex *vc = (Vertex*)target;
    Surface *s = (Surface*)source;

    FVector3 rel_c2ab;
    if(ConvexPolygonConstraint_acts(vc, s, rel_c2ab)) 
        e += vc->particle()->getMass() / _Engine.dt * lam / 2.0 * rel_c2ab.dot();

    return S_OK;
}

HRESULT ConvexPolygonConstraint::force(const MeshObj *source, const MeshObj *target, FloatP_t *f) {
    Vertex *vc = (Vertex*)target;
    Surface *s = (Surface*)source;

    FVector3 rel_c2ab;
    if(ConvexPolygonConstraint_acts(vc, s, rel_c2ab)) {
        const FVector3 force = rel_c2ab * vc->particle()->getMass() / _Engine.dt * lam;
        f[0] += force[0];
        f[1] += force[1];
        f[2] += force[2];
    }

    return S_OK;
}
