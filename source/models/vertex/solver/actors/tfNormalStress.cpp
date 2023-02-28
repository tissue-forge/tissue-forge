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

#include "tfNormalStress.h"

#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfBody.h>

#include <tfEngine.h>
#include <io/tfFIO.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


FloatP_t NormalStress::energy(const Surface *source, const Vertex *target) {
    FVector3 fv = force(source, target);
    return fv.dot(target->getVelocity()) * _Engine.dt;
}

FVector3 NormalStress::force(const Surface *source, const Vertex *target) {
    auto bodies = source->getBodies();
    if(bodies.size() == 2) 
        return FVector3(0);
    
    FVector3 snormal = source->getNormal();
    if(snormal.isZero()) 
        return FVector3(0);

    snormal = snormal.normalized();
    if(bodies.size() == 1) 
        snormal *= source->volumeSense(bodies[0]);

    return snormal * mag * source->getVertexArea(target);
}

namespace TissueForge::io { 


    template <>
    HRESULT toFile(NormalStress *dataElement, const MetaData &metaData, IOElement &fileElement) { 

        TF_IOTOEASY(fileElement, metaData, "mag", dataElement->mag);

        fileElement.get()->type = "NormalStress";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, NormalStress **dataElement) { 

        FloatP_t mag;
        TF_IOFROMEASY(fileElement, metaData, "mag", &mag);
        *dataElement = new NormalStress(mag);

        return S_OK;
    }

};

NormalStress *NormalStress::fromString(const std::string &str) {
    return TissueForge::io::fromString<NormalStress*>(str);
}
