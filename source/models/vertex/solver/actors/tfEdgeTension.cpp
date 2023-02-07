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
#include <io/tfFIO.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


static inline FloatP_t EdgeTension_energy_order1(const Surface *source, const Vertex *target, const FloatP_t &lam, const unsigned int &dummy) {
    Vertex *vp, *vn;
    std::tie(vp, vn) = source->neighborVertices(target);

    TissueForge::FVector3 posc = target->getPosition();
    
    FloatP_t _e = metrics::relativePosition(vp->getPosition(), posc).length();
    _e += metrics::relativePosition(posc, vn->getPosition()).length();

    return lam * _e;
}

static inline FloatP_t EdgeTension_energy_orderN(const Surface *source, const Vertex *target, const FloatP_t &lam, const unsigned int &order) {
    Vertex *vp, *vn;
    std::tie(vp, vn) = source->neighborVertices(target);

    TissueForge::FVector3 posc = target->getPosition();
    
    const FloatP_t _ep1 = metrics::relativePosition(vp->getPosition(), posc).length();
    const FloatP_t _en1 = metrics::relativePosition(posc, vn->getPosition()).length();
    FloatP_t _ep = _ep1;
    FloatP_t _en = _en1;
    for(size_t i = 0; i < order; i++) {
        _ep *= _ep1;
        _en *= _en1;
    }

    return lam * (_ep + _en);
}

static inline TissueForge::FVector3 EdgeTension_force_order1(const Surface *source, const Vertex *target, const FloatP_t &lam, const unsigned int &dummy) {
    Vertex *vp, *vn;
    std::tie(vp, vn) = source->neighborVertices(target);
    if(!vp || !vn) 
        return TissueForge::FVector3(0);

    TissueForge::FVector3 posc = target->getPosition();
    TissueForge::FVector3 force;

    const TissueForge::FVector3 relpos_p = metrics::relativePosition(vp->getPosition(), posc);
    if(!relpos_p.isZero()) 
        force += relpos_p.normalized();
    const TissueForge::FVector3 relpos_n = metrics::relativePosition(vn->getPosition(), posc);
    if(!relpos_n.isZero()) 
        force += relpos_n.normalized();

    return lam * force;
}

static inline TissueForge::FVector3 EdgeTension_force_orderN(const Surface *source, const Vertex *target, const FloatP_t &lam, const unsigned int &order) {
    Vertex *vp, *vn;
    std::tie(vp, vn) = source->neighborVertices(target);

    const TissueForge::FVector3 posc = target->getPosition();

    const TissueForge::FVector3 posc2p = metrics::relativePosition(vp->getPosition(), posc);
    const TissueForge::FVector3 posc2n = metrics::relativePosition(vn->getPosition(), posc);
    const FloatP_t lenc2p1 = posc2p.length();
    const FloatP_t lenc2n1 = posc2n.length();
    FloatP_t lenc2p = lenc2p1;
    FloatP_t lenc2n = lenc2n1;
    for(size_t i = 0; i < order - 2; i++) {
        lenc2p *= lenc2p1;
        lenc2n *= lenc2n1;
    }

    return (posc2p * lenc2p + posc2n * lenc2n) * lam * (FloatP_t)order;
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


FloatP_t EdgeTension::energy(const Surface *source, const Vertex *target) {
    return energyFcn(source, target, lam, order);
}

TissueForge::FVector3 EdgeTension::force(const Surface *source, const Vertex *target) {
    return forceFcn(source, target, lam, order);
}

namespace TissueForge::io { 


    #define TF_ACTORIOTOEASY(fe, key, member) \
        fe = new IOElement(); \
        if(toFile(member, metaData, fe) != S_OK)  \
            return E_FAIL; \
        fe->parent = fileElement; \
        fileElement->children[key] = fe;

    #define TF_ACTORIOFROMEASY(feItr, children, metaData, key, member_p) \
        feItr = children.find(key); \
        if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
            return E_FAIL;

    template <>
    HRESULT toFile(EdgeTension *dataElement, const MetaData &metaData, IOElement *fileElement) { 

        IOElement *fe;

        TF_ACTORIOTOEASY(fe, "lam", dataElement->lam);
        TF_ACTORIOTOEASY(fe, "order", dataElement->order);

        fileElement->type = "EdgeTension";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, EdgeTension **dataElement) { 

        IOChildMap::const_iterator feItr;

        FloatP_t lam;
        unsigned int order;
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "lam", &lam);
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "order", &order);
        *dataElement = new EdgeTension(lam, order);

        return S_OK;
    }

};

EdgeTension *EdgeTension::fromString(const std::string &str) {
    return TissueForge::io::fromString<EdgeTension*>(str);
}
