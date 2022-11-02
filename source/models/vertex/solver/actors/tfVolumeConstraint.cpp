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

#include "tfVolumeConstraint.h"

#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfBody.h>


using namespace TissueForge::models::vertex;


HRESULT VolumeConstraint::energy(const MeshObj *source, const MeshObj *target, FloatP_t &e) {
    Body *b = (Body*)source;
    FloatP_t dvol = b->getVolume() - constr;
    e = lam * dvol * dvol;
    return S_OK;
}

HRESULT VolumeConstraint::force(const MeshObj *source, const MeshObj *target, FloatP_t *f) {
    Body *b = (Body*)source;
    Vertex *v = (Vertex*)target;
    
    FVector3 posc = v->getPosition();
    FVector3 ftotal(0.f);

    Vertex *vp, *vn;

    for(auto &s : v->getSurfaces()) {
        if(!s->in(b)) 
            continue;
        
        auto svertices = s->getVertices();
        std::tie(vp, vn) = s->neighborVertices(v);

        FVector3 sftotal = Magnum::Math::cross(s->getCentroid(), vp->getPosition() - vn->getPosition());
        for(unsigned int i = 0; i < svertices.size(); i++) {
            sftotal -= s->triangleNormal(i) / svertices.size();
        }

        ftotal += sftotal * s->volumeSense(b);
    }
    
    ftotal *= (lam * (b->getVolume() - constr) / 3.f);

    f[0] += ftotal[0];
    f[1] += ftotal[1];
    f[2] += ftotal[2];

    return S_OK;
}
