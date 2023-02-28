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
#include <tf_metrics.h>
#include <io/tfFIO.h>

#include <Magnum/Math/Math.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


static inline bool ConvexPolygonConstraint_acts(const Vertex *vc, const Surface *s, FVector3 &rel_c2ab) {
    std::vector<Vertex*> vertices = s->getVertices();
    if(vertices.size() <= 3) 
        return false;

    Vertex *va, *vb;
    std::tie(va, vb) = s->neighborVertices(vc);
    const FVector3 scent = s->getCentroid();
    const FVector3 posva = scent + metrics::relativePosition(va->getPosition(), scent);
    const FVector3 posvb = scent + metrics::relativePosition(vb->getPosition(), scent);
    const FVector3 posvc = scent + metrics::relativePosition(vc->getPosition(), scent);

    const FloatP_t areaABc = Magnum::Math::cross(metrics::relativePosition(posva, scent), metrics::relativePosition(posvb, scent)).length();
    const FloatP_t areaABC = Magnum::Math::cross(metrics::relativePosition(posva, posvc), metrics::relativePosition(posvb, posvc)).length();
    const FloatP_t areaBcC = Magnum::Math::cross(metrics::relativePosition(posvb, posvc), metrics::relativePosition(scent, posvc)).length();
    const FloatP_t areacAC = Magnum::Math::cross(metrics::relativePosition(scent, posvc), metrics::relativePosition(posva, posvc)).length();

    if(abs((areaABC + areaBcC + areacAC) / areaABc - 1) >= 1E-6) 
        return false;

    FVector3 lineDir = posvb - posva;
    if(lineDir.isZero()) 
        return false;
    lineDir = lineDir.normalized();

    rel_c2ab = posva + (posvc - posva).dot(lineDir) * lineDir - posvc;
    
    return true;
}

FloatP_t ConvexPolygonConstraint::energy(const Surface *source, const Vertex *target) {
    FloatP_t e;
    FVector3 rel_c2ab;
    if(ConvexPolygonConstraint_acts(target, source, rel_c2ab)) 
        e += target->getCachedParticleMass() / _Engine.dt * lam / 2.0 * rel_c2ab.dot();

    return e;
}

FVector3 ConvexPolygonConstraint::force(const Surface *source, const Vertex *target) {
    FVector3 force;
    FVector3 rel_c2ab;
    if(ConvexPolygonConstraint_acts(target, source, rel_c2ab)) {
        force += rel_c2ab * target->getCachedParticleMass() / _Engine.dt * lam;
    }

    return force;
}

namespace TissueForge::io { 


    template <>
    HRESULT toFile(ConvexPolygonConstraint *dataElement, const MetaData &metaData, IOElement &fileElement) { 

        TF_IOTOEASY(fileElement, metaData, "lam", dataElement->lam);

        fileElement.get()->type = "ConvexPolygonConstraint";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, ConvexPolygonConstraint **dataElement) { 

        FloatP_t lam;
        TF_IOFROMEASY(fileElement, metaData, "lam", &lam);
        *dataElement = new ConvexPolygonConstraint(lam);

        return S_OK;
    }

};

ConvexPolygonConstraint *ConvexPolygonConstraint::fromString(const std::string &str) {
    return TissueForge::io::fromString<ConvexPolygonConstraint*>(str);
}
