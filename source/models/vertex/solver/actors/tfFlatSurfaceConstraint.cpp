/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego and Tien Comlekoglu
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

#include "tfFlatSurfaceConstraint.h"

#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfVertex.h>

#include <tfEngine.h>
#include <io/tfFIO.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


FloatP_t FlatSurfaceConstraint::energy(const Surface *source, const Vertex *target) {
    FloatP_t _e = (source->getCentroid() - target->getPosition()).dot(source->getNormal());
    
    return target->getCachedParticleMass() / 2.f / _Engine.dt * lam * _e * _e;
}

FVector3 FlatSurfaceConstraint::force(const Surface *source, const Vertex *target) {
    FVector3 sn = source->getNormal();
    
    return (sn * ((source->getCentroid() - target->getPosition()).dot(sn))) * target->getCachedParticleMass() / _Engine.dt * lam;
}

namespace TissueForge::io { 


    template <>
    HRESULT toFile(FlatSurfaceConstraint *dataElement, const MetaData &metaData, IOElement &fileElement) { 

        TF_IOTOEASY(fileElement, metaData, "lam", dataElement->lam);

        fileElement.get()->type = "FlatSurfaceConstraint";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, FlatSurfaceConstraint **dataElement) { 

        FloatP_t lam;
        TF_IOFROMEASY(fileElement, metaData, "lam", &lam);
        *dataElement = new FlatSurfaceConstraint(lam);

        return S_OK;
    }

};

FlatSurfaceConstraint *FlatSurfaceConstraint::fromString(const std::string &str) {
    return TissueForge::io::fromString<FlatSurfaceConstraint*>(str);
}
