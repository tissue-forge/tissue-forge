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

#include "tfAdhesion.h"

#include <models/vertex/solver/tfBody.h>
#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfVertex.h>

#include <types/tf_types.h>
#include <tf_metrics.h>
#include <io/tfFIO.h>

#include <Magnum/Math/Math.h>

#include <unordered_set>
#include <vector>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


static HRESULT Adhesion_energy_Body(Body *b, Vertex *v, const FloatP_t &lam, const std::unordered_set<int> &targetTypes, FloatP_t &e) {
    FloatP_t _e = 0.0;

    fVector3 posv = v->getPosition();

    for(auto &s : v->getSurfaces()) {
        std::vector<Body*> bodies = s->getBodies();
        Body *b1 = bodies[0];
        Body *b2 = bodies[1];
        Body *bo = NULL;
        if(b1 == b) 
            bo = b2;
        else if(b2 == b) 
            bo = b1;
        
        if(!bo || targetTypes.find(bo->typeId) == targetTypes.end()) 
            continue;

        Vertex *vp = std::get<0>(s->neighborVertices(v));

        _e += lam * metrics::relativePosition(vp->getPosition(), posv).length();
    }

    e += 0.5 * _e;

    return S_OK;
}

static HRESULT Adhesion_force_Body(Body *b, Vertex *v, const FloatP_t &lam, const std::unordered_set<int> &targetTypes, FloatP_t *f) {
    fVector3 _f(0.0);

    fVector3 posv = v->getPosition();

    Vertex *vp, *vn;

    for(auto &s : v->getSurfaces()) {
        std::vector<Body*> bodies = s->getBodies();
        Body *b1 = bodies[0];
        Body *b2 = bodies[1];
        Body *bo = NULL;
        if(b1 == b) 
            bo = b2;
        else if(b2 == b) 
            bo = b1;
        
        if(!bo || targetTypes.find(bo->typeId) == targetTypes.end()) 
            continue;

        fVector3 scent = s->getCentroid();
        std::tie(vp, vn) = s->neighborVertices(v);
        fVector3 posvp = vp->getPosition();
        fVector3 posvn = vn->getPosition();
        fVector3 posv_rel = posv - scent;

        fVector3 normvp = Magnum::Math::cross(posv_rel, posvp - scent);
        fVector3 normvn = Magnum::Math::cross(posvn - scent, posv_rel);

        if(!normvp.isZero() && !normvn.isZero()) 
            _f += (1.0 / s->getVertices().size() - 1.0) * (Magnum::Math::cross(normvp.normalized(), posv - posvp) + 
                Magnum::Math::cross(normvn.normalized(), posvn - posv));
    }

    FloatP_t fact = 0.5 * lam;
    f[0] += fact * _f[0];
    f[1] += fact * _f[1];
    f[2] += fact * _f[2];

    return S_OK;
}

static HRESULT Adhesion_energy_Surface(Surface *s, Vertex *v, const FloatP_t &lam, const std::unordered_set<int> &targetTypes, FloatP_t &e) {
    FloatP_t _e = 0.0;
    fVector3 posv = v->getPosition();

    for(auto &sv : v->getSurfaces()) 
        if(sv != s && targetTypes.find(sv->typeId) != targetTypes.end()) 
            _e += metrics::relativePosition(std::get<0>(sv->neighborVertices(v))->getPosition(), posv).length();

    e += lam * _e;
    return S_OK;
}

static HRESULT Adhesion_force_Surface(Surface *s, Vertex *v, const FloatP_t &lam, const std::unordered_set<int> &targetTypes, FloatP_t *f) {
    fVector3 _f(0.0);
    fVector3 posv = v->getPosition();

    Vertex *vp, *vn;

    for(auto &sv : v->getSurfaces()) 
        if(sv != s && targetTypes.find(sv->typeId) != targetTypes.end()) {
            std::tie(vp, vn) = s->neighborVertices(v);
            fVector3 posvp_rel = metrics::relativePosition(vp->getPosition(), posv);
            fVector3 posvn_rel = metrics::relativePosition(vn->getPosition(), posv);
            if(!posvp_rel.isZero() && !posvn_rel.isZero())
                _f += posvp_rel.normalized() + posvn_rel.normalized();
        }

    FloatP_t fact = 0.5 * lam;
    f[0] += fact * _f[0];
    f[1] += fact * _f[1];
    f[2] += fact * _f[2];

    return S_OK;
}


HRESULT Adhesion::energy(const MeshObj *source, const MeshObj *target, FloatP_t &e) {
    if(source->objType() == MeshObj::Type::BODY) { 
        Body *b = (Body*)source;
        auto itr = typePairs.find(b->typeId);
        if(itr == typePairs.end()) 
            return S_OK;
        return Adhesion_energy_Body(b, (Vertex*)target, lam, itr->second, e);
    }
    else if(source->objType() == MeshObj::Type::SURFACE) {
        Surface *s = (Surface*)source;
        auto itr = typePairs.find(s->typeId);
        if(itr == typePairs.end()) 
            return S_OK;
        return Adhesion_energy_Surface(s, (Vertex*)target, lam, itr->second, e);
    }
    return S_OK;
}

HRESULT Adhesion::force(const MeshObj *source, const MeshObj *target, FloatP_t *f) {
    if(source->objType() == MeshObj::Type::BODY) { 
        Body *b = (Body*)source;
        auto itr = typePairs.find(b->typeId);
        if(itr == typePairs.end()) 
            return S_OK;
        return Adhesion_force_Body(b, (Vertex*)target, lam, itr->second, f);
    }
    else if(source->objType() == MeshObj::Type::SURFACE) {
        Surface *s = (Surface*)source;
        auto itr = typePairs.find(s->typeId);
        if(itr == typePairs.end()) 
            return S_OK;
        return Adhesion_force_Surface(s, (Vertex*)target, lam, itr->second, f);
    }
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
    HRESULT toFile(Adhesion *dataElement, const MetaData &metaData, IOElement *fileElement) { 

        IOElement *fe;

        TF_ACTORIOTOEASY(fe, "lam", dataElement->lam);

        fileElement->type = "Adhesion";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Adhesion **dataElement) { 

        IOChildMap::const_iterator feItr;

        FloatP_t lam;
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "lam", &lam);
        *dataElement = new Adhesion(lam);

        return S_OK;
    }

};

Adhesion *Adhesion::fromString(const std::string &str) {
    return TissueForge::io::fromString<Adhesion*>(str);
}
