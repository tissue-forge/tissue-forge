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

#include "tfEdgeTension.h"

#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfVertex.h>

#include <tf_metrics.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


HRESULT EdgeTension::energy(MeshObj *source, MeshObj *target, float &e) {
    Surface *s = (Surface*)source;
    Vertex *v = (Vertex*)target;

    std::vector<Vertex*> vertices = s->getVertices();
    std::vector<Vertex*>::iterator vc_itr = std::find(vertices.begin(), vertices.end(), v);
    if(vc_itr == vertices.end()) 
        return S_OK;

    Vertex *vn = vc_itr == vertices.begin()   ? *(vertices.end() - 1) : *(vc_itr - 1);
    Vertex *vp = vc_itr == vertices.end() - 1 ? *(vertices.begin())   : *(vc_itr + 1);

    FVector3 posc = v->getPosition();
    
    float _e = metrics::relativePosition(vp->getPosition(), posc).length();
    _e += metrics::relativePosition(posc, vn->getPosition()).length();

    e += lam * _e;
    return S_OK;
}

HRESULT EdgeTension::force(MeshObj *source, MeshObj *target, float *f) {
    Surface *s = (Surface*)source;
    Vertex *v = (Vertex*)target;

    std::vector<Vertex*> vertices = s->getVertices();
    std::vector<Vertex*>::iterator vc_itr = std::find(vertices.begin(), vertices.end(), v);
    if(vc_itr == vertices.end()) 
        return S_OK;

    Vertex *vn = vc_itr == vertices.begin()   ? *(vertices.end() - 1) : *(vc_itr - 1);
    Vertex *vp = vc_itr == vertices.end() - 1 ? *(vertices.begin())   : *(vc_itr + 1);

    FVector3 posc = v->getPosition();
    FVector3 force;

    force = (metrics::relativePosition(vp->getPosition(), posc).normalized() + metrics::relativePosition(vn->getPosition(), posc).normalized()) * lam;

    f[0] += force[0];
    f[1] += force[1];
    f[2] += force[2];
    
    return S_OK;
}
