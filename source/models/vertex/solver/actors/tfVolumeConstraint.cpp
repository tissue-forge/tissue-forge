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
        
        std::tie(vp, vn) = s->neighborVertices(target);

        ftotal += (
            Magnum::Math::cross(s->getCentroid(), vn->getPosition() - vp->getPosition()) + 
            s->getUnnormalizedNormal() / s->getVertices().size()
        ) * s->volumeSense(source);
    }
    
    return ftotal * (lam * (constr - source->getVolume()) / 3.f);
}

namespace TissueForge::io { 


    template <>
    HRESULT toFile(VolumeConstraint *dataElement, const MetaData &metaData, IOElement &fileElement) { 

        TF_IOTOEASY(fileElement, metaData, "lam", dataElement->lam);
        TF_IOTOEASY(fileElement, metaData, "constr", dataElement->constr);

        fileElement.get()->type = "VolumeConstraint";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, VolumeConstraint **dataElement) { 

        FloatP_t lam, constr;
        TF_IOFROMEASY(fileElement, metaData, "lam", &lam);
        TF_IOFROMEASY(fileElement, metaData, "constr", &constr);
        *dataElement = new VolumeConstraint(lam, constr);

        return S_OK;
    }

};

VolumeConstraint *VolumeConstraint::fromString(const std::string &str) {
    return TissueForge::io::fromString<VolumeConstraint*>(str);
}
