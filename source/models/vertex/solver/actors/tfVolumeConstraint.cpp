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
#include <models/vertex/solver/tf_mesh_io.h>

#include <io/tfFIO.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


FloatP_t VolumeConstraint::energy(const Body *source, const Vertex *target) {
    FloatP_t dvol = source->getVolume() - constr;
    return lam * dvol * dvol;
}

FVector3 VolumeConstraint::force(const Body *source, const Vertex *target) {
    
    FVector3 posc = target->getPosition();
    FVector3 ftotal(0.f);

    Vertex *vp, *vn;

    for(auto &s : target->getSurfaces()) {
        if(!s->defines(source)) 
            continue;
        
        auto svertices = s->getVertices();
        std::tie(vp, vn) = s->neighborVertices(target);

        FVector3 sftotal = Magnum::Math::cross(s->getCentroid(), vp->getPosition() - vn->getPosition());
        for(unsigned int i = 0; i < svertices.size(); i++) {
            sftotal -= s->triangleNormal(i) / svertices.size();
        }

        ftotal += sftotal * s->volumeSense(source);
    }
    
    return ftotal * (lam * (source->getVolume() - constr) / 3.f);
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
    HRESULT toFile(VolumeConstraint *dataElement, const MetaData &metaData, IOElement *fileElement) { 

        IOElement *fe;

        TF_ACTORIOTOEASY(fe, "lam", dataElement->lam);
        TF_ACTORIOTOEASY(fe, "constr", dataElement->constr);

        fileElement->type = "VolumeConstraint";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, VolumeConstraint **dataElement) { 

        IOChildMap::const_iterator feItr;

        FloatP_t lam, constr;
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "lam", &lam);
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "constr", &constr);
        *dataElement = new VolumeConstraint(lam, constr);

        return S_OK;
    }

};

VolumeConstraint *VolumeConstraint::fromString(const std::string &str) {
    return TissueForge::io::fromString<VolumeConstraint*>(str);
}
