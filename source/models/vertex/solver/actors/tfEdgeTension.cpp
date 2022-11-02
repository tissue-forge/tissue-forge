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
#include <tfError.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


static inline HRESULT EdgeTension_energy_order1(MeshObj *source, MeshObj *target, const FloatP_t &lam, const unsigned int &dummy, FloatP_t &e) {
    Surface *s = (Surface*)source;
    Vertex *v = (Vertex*)target;

    Vertex *vp, *vn;
    std::tie(vp, vn) = s->neighborVertices(v);
    if(!vp || !vn) 
        return S_OK;

    FVector3 posc = v->getPosition();
    
    FloatP_t _e = metrics::relativePosition(vp->getPosition(), posc).length();
    _e += metrics::relativePosition(posc, vn->getPosition()).length();

    e += lam * _e;
    return S_OK;
}

static inline HRESULT EdgeTension_energy_orderN(MeshObj *source, MeshObj *target, const FloatP_t &lam, const unsigned int &order, FloatP_t &e) {
    Surface *s = (Surface*)source;
    Vertex *v = (Vertex*)target;

    Vertex *vp, *vn;
    std::tie(vp, vn) = s->neighborVertices(v);
    if(!vp || !vn) 
        return S_OK;

    FVector3 posc = v->getPosition();
    
    const FloatP_t _ep1 = metrics::relativePosition(vp->getPosition(), posc).length();
    const FloatP_t _en1 = metrics::relativePosition(posc, vn->getPosition()).length();
    FloatP_t _ep = _ep1;
    FloatP_t _en = _en1;
    for(size_t i = 0; i < order; i++) {
        _ep *= _ep1;
        _en *= _en1;
    }

    e += lam * (_ep + _en);
    return S_OK;
}

static inline HRESULT EdgeTension_force_order1(MeshObj *source, MeshObj *target, const FloatP_t &lam, const unsigned int &dummy, FloatP_t *f) {
    Surface *s = (Surface*)source;
    Vertex *v = (Vertex*)target;

    Vertex *vp, *vn;
    std::tie(vp, vn) = s->neighborVertices(v);
    if(!vp || !vn) 
        return S_OK;

    FVector3 posc = v->getPosition();
    FVector3 force;

    force = (metrics::relativePosition(vp->getPosition(), posc).normalized() + metrics::relativePosition(vn->getPosition(), posc).normalized()) * lam;

    f[0] += force[0];
    f[1] += force[1];
    f[2] += force[2];
    
    return S_OK;
}

static inline HRESULT EdgeTension_force_orderN(MeshObj *source, MeshObj *target, const FloatP_t &lam, const unsigned int &order, FloatP_t *f) {
    Surface *s = (Surface*)source;
    Vertex *v = (Vertex*)target;

    Vertex *vp, *vn;
    std::tie(vp, vn) = s->neighborVertices(v);
    if(!vp || !vn) 
        return S_OK;

    const FVector3 posc = v->getPosition();

    const FVector3 posc2p = metrics::relativePosition(vp->getPosition(), posc);
    const FVector3 posc2n = metrics::relativePosition(vn->getPosition(), posc);
    const FloatP_t lenc2p1 = posc2p.length();
    const FloatP_t lenc2n1 = posc2n.length();
    FloatP_t lenc2p = lenc2p1;
    FloatP_t lenc2n = lenc2n1;
    for(size_t i = 0; i < order - 2; i++) {
        lenc2p *= lenc2p1;
        lenc2n *= lenc2n1;
    }

    const FVector3 force = (posc2p * lenc2p + posc2n * lenc2n) * lam * (FloatP_t)order;

    f[0] += force[0];
    f[1] += force[1];
    f[2] += force[2];
    
    return S_OK;
}


EdgeTension::EdgeTension(const FloatP_t &_lam, const unsigned int &_order) {
    lam = _lam;
    order = _order;
    
    if(order < 1) {
        tf_error(E_FAIL, "Edge tension order must be greater than 0. Defaulting to 1");
        order = 1;
    }

    if(order == 1) {
        energyFcn = EdgeTension_energy_order1;
        forceFcn = EdgeTension_force_order1;
    } 
    else {
        energyFcn = EdgeTension_energy_orderN;
        forceFcn = EdgeTension_force_orderN;
    }
}


HRESULT EdgeTension::energy(MeshObj *source, MeshObj *target, FloatP_t &e) {
    return energyFcn(source, target, lam, order, e);
}

HRESULT EdgeTension::force(MeshObj *source, MeshObj *target, FloatP_t *f) {
    return forceFcn(source, target, lam, order, f);
}
