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

#include "tfPerimeterConstraint.h"

#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfBody.h>

#include <io/tfFIO.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;

FloatP_t PerimeterConstraint::energy(const Surface *source, const Vertex *target) {
    FloatP_t dper = source->getPerimeter() - constr;
    return lam * dper * dper;
}

FVector3 PerimeterConstraint::force(const Surface *source, const Vertex *target) {
    Vertex *vp, *vn;
    std::tie(vp, vn) = source->neighborVertices(target);
    const FVector3 posc = target->getPosition();
    const FVector3 posp_rel = vp->getPosition() - posc;
    const FVector3 posn_rel = vn->getPosition() - posc;
    FVector3 result(0.f);
    if(!posp_rel.isZero()) 
        result += posp_rel.normalized();
    if(!posn_rel.isZero()) 
        result += posn_rel.normalized();
    return result * 2.0 * lam * (source->getPerimeter() - constr);
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
    HRESULT toFile(PerimeterConstraint *dataElement, const MetaData &metaData, IOElement *fileElement) { 

        IOElement *fe;

        TF_ACTORIOTOEASY(fe, "lam", dataElement->lam);
        TF_ACTORIOTOEASY(fe, "constr", dataElement->constr);

        fileElement->type = "PerimeterConstraint";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, PerimeterConstraint **dataElement) { 

        IOChildMap::const_iterator feItr;

        FloatP_t lam, constr;
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "lam", &lam);
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "constr", &constr);
        *dataElement = new PerimeterConstraint(lam, constr);

        return S_OK;
    }

};

PerimeterConstraint *PerimeterConstraint::fromString(const std::string &str) {
    return TissueForge::io::fromString<PerimeterConstraint*>(str);
}
