/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego and Tien Comlekoglu
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

#include "tfSurfaceAreaConstraint.h"

#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfBody.h>

#include <io/tfFIO.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


FloatP_t SurfaceAreaConstraint::energy(const Body *source, const Vertex *target) {
    FloatP_t darea = source->getArea() - constr;
    return lam * darea * darea;
}

FVector3 SurfaceAreaConstraint::force(const Body *source, const Vertex *target) {
    FVector3 ftotal;

    for(auto &s : target->getSurfaces()) {
        if(!s->defines(source)) 
            continue;
        
        std::vector<Vertex*> svertices = s->getVertices();
        const FVector3 scent = s->getCentroid();
        for(std::vector<Vertex*>::iterator itr = svertices.begin(); itr != svertices.end(); itr++) {
            Vertex *vc = *itr;
            Vertex *vn = itr + 1 == svertices.end() ? svertices.front() : *(itr + 1);
            const FVector3 posvc = vc->getPosition();
            const FVector3 posvn = vn->getPosition();
            const FVector3 triNorm = Magnum::Math::cross(posvc - scent, posvn - scent);
            if(triNorm.isZero()) 
                continue;
            FVector3 g = (posvc - posvn) / svertices.size();
            if(vc == target) 
                g += posvn - scent;
            else if(vn == target) 
                g -= posvc - scent;
            ftotal += Magnum::Math::cross(triNorm.normalized(), g);
        }
    }

    return ftotal * (lam * (source->getArea() - constr));
}

FloatP_t SurfaceAreaConstraint::energy(const Surface *source, const Vertex *target) {
    FloatP_t darea = source->getArea() - constr;
    return lam * darea * darea;
}

FVector3 SurfaceAreaConstraint::force(const Surface *source, const Vertex *target) {
    FVector3 ftotal;

    auto svertices = source->getVertices();
    unsigned int idx, idxc, idxp, idxn;
    FVector3 sftotal, triNorm;
    
    for(idx = 0; idx < svertices.size(); idx++) {
        if(svertices[idx] == target) 
            idxc = idx;
        triNorm = source->triangleNormal(idx);
        if(!triNorm.isZero())
            sftotal += Magnum::Math::cross(
                triNorm.normalized(), 
                svertices[idx == svertices.size() - 1 ? 0 : idx + 1]->getPosition() - svertices[idx]->getPosition()
            );
    }
    sftotal /= svertices.size();

    idxp = idxc == 0 ? svertices.size() - 1 : idxc - 1;
    idxn = idxc == svertices.size() - 1 ? 0 : idxc + 1;

    const FVector3 scentroid = source->getCentroid();

    triNorm = source->triangleNormal(idxc);
    if(!triNorm.isZero()) 
        sftotal += Magnum::Math::cross(triNorm.normalized(), scentroid - svertices[idxn]->getPosition());
    triNorm = source->triangleNormal(idxp);
    if(!triNorm.isZero()) 
        sftotal -= Magnum::Math::cross(triNorm.normalized(), scentroid - svertices[idxp]->getPosition());
    ftotal += sftotal;

    ftotal *= (lam * (constr - source->getArea()));

    return ftotal;
}

namespace TissueForge::io { 


    template <>
    HRESULT toFile(SurfaceAreaConstraint *dataElement, const MetaData &metaData, IOElement &fileElement) { 

        TF_IOTOEASY(fileElement, metaData, "lam", dataElement->lam);
        TF_IOTOEASY(fileElement, metaData, "constr", dataElement->constr);

        fileElement.get()->type = "SurfaceAreaConstraint";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, SurfaceAreaConstraint **dataElement) { 

        FloatP_t lam, constr;
        TF_IOFROMEASY(fileElement, metaData, "lam", &lam);
        TF_IOFROMEASY(fileElement, metaData, "constr", &constr);
        *dataElement = new SurfaceAreaConstraint(lam, constr);

        return S_OK;
    }

};

SurfaceAreaConstraint *SurfaceAreaConstraint::fromString(const std::string &str) {
    return TissueForge::io::fromString<SurfaceAreaConstraint*>(str);
}
