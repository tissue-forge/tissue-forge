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

#include "tfSurfaceAreaConstraint.h"

#include <models/vertex/solver/tfVertex.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfBody.h>

#include <io/tfFIO.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


static HRESULT SurfaceAreaConstraint_energy_Body(Body *b, const FloatP_t &lam, const FloatP_t &constr, FloatP_t &e) {
    FloatP_t darea = b->getArea() - constr;
    e = lam * darea * darea;
    return S_OK;
}

static HRESULT SurfaceAreaConstraint_force_Body(Body *b, Vertex *v, const FloatP_t &lam, const FloatP_t &constr, FloatP_t *f) {
    FVector3 ftotal;

    for(auto &s : v->getSurfaces()) {
        if(!s->in(b)) 
            continue;
        
        auto svertices = s->getVertices();
        unsigned int idx, idxc, idxp, idxn;
        FVector3 sftotal, triNorm;
        
        for(idx = 0; idx < svertices.size(); idx++) {
            if(svertices[idx] == v) 
                idxc = idx;
            triNorm = s->triangleNormal(idx);
            if(!triNorm.isZero()) 
                sftotal += Magnum::Math::cross(
                    triNorm.normalized(), 
                    svertices[idx == svertices.size() - 1 ? 0 : idx + 1]->getPosition() - svertices[idx]->getPosition()
                );
        }
        sftotal /= svertices.size();

        idxp = idxc == 0 ? svertices.size() - 1 : idxc - 1;
        idxn = idxc == svertices.size() - 1 ? 0 : idxc + 1;

        const FVector3 scentroid = s->getCentroid();

        triNorm = s->triangleNormal(idxc);
        if(!triNorm.isZero()) 
            sftotal += Magnum::Math::cross(triNorm.normalized(), scentroid - svertices[idxn]->getPosition());
        triNorm = s->triangleNormal(idxp);
        if(!triNorm.isZero()) 
            sftotal -= Magnum::Math::cross(triNorm.normalized(), scentroid - svertices[idxp]->getPosition());
        ftotal += sftotal;
    }

    ftotal *= (lam * (constr - b->getArea()));

    f[0] += ftotal[0];
    f[1] += ftotal[1];
    f[2] += ftotal[2];

    return S_OK;
}

static HRESULT SurfaceAreaConstraint_energy_Surface(Surface *s, const FloatP_t &lam, const FloatP_t &constr, FloatP_t &e) {
    FloatP_t darea = s->getArea() - constr;
    e = lam * darea * darea;
    return S_OK;
}

static HRESULT SurfaceAreaConstraint_force_Surface(Surface *s, Vertex *v, const FloatP_t &lam, const FloatP_t &constr, FloatP_t *f) {
    FVector3 ftotal;

    auto svertices = s->getVertices();
    unsigned int idx, idxc, idxp, idxn;
    FVector3 sftotal, triNorm;
    
    for(idx = 0; idx < svertices.size(); idx++) {
        if(svertices[idx] == v) 
            idxc = idx;
        triNorm = s->triangleNormal(idx);
        if(!triNorm.isZero())
            sftotal += Magnum::Math::cross(
                triNorm.normalized(), 
                svertices[idx == svertices.size() - 1 ? 0 : idx + 1]->getPosition() - svertices[idx]->getPosition()
            );
    }
    sftotal /= svertices.size();

    idxp = idxc == 0 ? svertices.size() - 1 : idxc - 1;
    idxn = idxc == svertices.size() - 1 ? 0 : idxc + 1;

    const FVector3 scentroid = s->getCentroid();

    triNorm = s->triangleNormal(idxc);
    if(!triNorm.isZero()) 
        sftotal += Magnum::Math::cross(triNorm.normalized(), scentroid - svertices[idxn]->getPosition());
    triNorm = s->triangleNormal(idxp);
    if(!triNorm.isZero()) 
        sftotal -= Magnum::Math::cross(triNorm.normalized(), scentroid - svertices[idxp]->getPosition());
    ftotal += sftotal;

    ftotal *= (lam * (constr - s->getArea()));

    f[0] += ftotal[0];
    f[1] += ftotal[1];
    f[2] += ftotal[2];

    return S_OK;
}


HRESULT SurfaceAreaConstraint::energy(const MeshObj *source, const MeshObj *target, FloatP_t &e) {
    if(source->objType() == MeshObj::Type::BODY) 
        return SurfaceAreaConstraint_energy_Body((Body*)source, lam, constr, e);
    else if(source->objType() == MeshObj::Type::SURFACE) 
        return SurfaceAreaConstraint_energy_Surface((Surface*)source, lam, constr, e);
    return S_OK;
}

HRESULT SurfaceAreaConstraint::force(const MeshObj *source, const MeshObj *target, FloatP_t *f) {
    if(source->objType() == MeshObj::Type::BODY) 
        return SurfaceAreaConstraint_force_Body((Body*)source, (Vertex*)target, lam, constr, f);
    else if(source->objType() == MeshObj::Type::SURFACE) 
        return SurfaceAreaConstraint_force_Surface((Surface*)source, (Vertex*)target, lam, constr, f);
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
    HRESULT toFile(SurfaceAreaConstraint *dataElement, const MetaData &metaData, IOElement *fileElement) { 

        IOElement *fe;

        TF_ACTORIOTOEASY(fe, "lam", dataElement->lam);
        TF_ACTORIOTOEASY(fe, "constr", dataElement->constr);

        fileElement->type = "SurfaceAreaConstraint";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, SurfaceAreaConstraint **dataElement) { 

        IOChildMap::const_iterator feItr;

        FloatP_t lam, constr;
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "lam", &lam);
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "constr", &constr);
        *dataElement = new SurfaceAreaConstraint(lam, constr);

        return S_OK;
    }

};

SurfaceAreaConstraint *SurfaceAreaConstraint::fromString(const std::string &str) {
    return TissueForge::io::fromString<SurfaceAreaConstraint*>(str);
}
