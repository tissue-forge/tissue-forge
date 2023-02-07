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
    HRESULT toFile(BodyForce *dataElement, const MetaData &metaData, IOElement *fileElement) { 

        IOElement *fe;

        TF_ACTORIOTOEASY(fe, "comps", dataElement->comps);

        fileElement->type = "BodyForce";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, BodyForce **dataElement) { 

        IOChildMap::const_iterator feItr;

        FVector3 comps;
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "comps", &comps);
        *dataElement = new BodyForce(comps);

        return S_OK;
    }

};

BodyForce *BodyForce::fromString(const std::string &str) {
    return TissueForge::io::fromString<BodyForce*>(str);
}
