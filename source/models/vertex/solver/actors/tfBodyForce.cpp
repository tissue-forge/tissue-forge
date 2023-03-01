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

#include "tfBodyForce.h"

#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfBody.h>

#include <tfEngine.h>
#include <io/tfFIO.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


FloatP_t BodyForce::energy(const Body *source, const Vertex *target) {
    return force(source, target).dot(target->getVelocity()) * _Engine.dt;
}

FVector3 BodyForce::force(const Body *source, const Vertex *target) {
    FloatP_t bArea = source->getArea();
    return bArea == 0.f ? FVector3(0) : comps * source->getVertexArea(target) / bArea;
}

namespace TissueForge::io { 


    template <>
    HRESULT toFile(BodyForce *dataElement, const MetaData &metaData, IOElement &fileElement) { 

        TF_IOTOEASY(fileElement, metaData, "comps", dataElement->comps);

        fileElement.get()->type = "BodyForce";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, BodyForce **dataElement) { 

        FVector3 comps;
        TF_IOFROMEASY(fileElement, metaData, "comps", &comps);
        *dataElement = new BodyForce(comps);

        return S_OK;
    }

};

BodyForce *BodyForce::fromString(const std::string &str) {
    return TissueForge::io::fromString<BodyForce*>(str);
}
