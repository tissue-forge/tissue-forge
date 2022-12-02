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

#include "tfSurfaceTraction.h"

#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfSurface.h>

#include <tfEngine.h>
#include <io/tfFIO.h>


using namespace TissueForge::models::vertex;


HRESULT SurfaceTraction::energy(const MeshObj *source, const MeshObj *target, FloatP_t &e) {
    FVector3 fv;
    force(source, target, fv.data());
    e = fv.dot(((Vertex*)target)->particle()->getVelocity()) * _Engine.dt;
    return S_OK;
}

HRESULT SurfaceTraction::force(const MeshObj *source, const MeshObj *target, FloatP_t *f) {
    Surface *s = (Surface*)source;
    Vertex *v = (Vertex*)target;
    FVector3 vForce = comps * s->getVertexArea(v);
    f[0] += vForce[0];
    f[1] += vForce[1];
    f[2] += vForce[2];
    return S_OK;
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
    HRESULT toFile(SurfaceTraction *dataElement, const MetaData &metaData, IOElement *fileElement) { 

        IOElement *fe;

        TF_ACTORIOTOEASY(fe, "comps", dataElement->comps);

        fileElement->type = "SurfaceTraction";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, SurfaceTraction **dataElement) { 

        IOChildMap::const_iterator feItr;

        FVector3 comps;
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "comps", &comps);
        *dataElement = new SurfaceTraction(comps);

        return S_OK;
    }

};

SurfaceTraction *SurfaceTraction::fromString(const std::string &str) {
    return TissueForge::io::fromString<SurfaceTraction*>(str);
}
