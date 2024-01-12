/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego and Tien Comlekoglu
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
#include <models/vertex/solver/tfMeshSolver.h>
#include <models/vertex/solver/tfVertexSolverFIO.h>

#include <types/tf_types.h>
#include <tf_metrics.h>
#include <io/tfFIO.h>

#include <Magnum/Math/Math.h>

#include <unordered_set>
#include <vector>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


static FloatP_t Adhesion_energy_Body(const Body *b, const Vertex *v, const FloatP_t &lam, const std::unordered_set<int> &targetTypes) {
    FloatP_t e = 0.0;

    FVector3 posv = v->getPosition();

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

        e += metrics::relativePosition(vp->getPosition(), posv).length();
    }

    return 0.5 * lam * e;
}

static FVector3 Adhesion_force_Body(const Body *b, const Vertex *v, const FloatP_t &lam, const std::unordered_set<int> &targetTypes) {
    FVector3 f(0.0);

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

        std::vector<Vertex*> svertices = s->getVertices();
        const FVector3 scent = s->getCentroid();
        for(std::vector<Vertex*>::iterator itr = svertices.begin(); itr != svertices.end(); itr++) {
            Vertex *vc = *itr;
            Vertex *vn = itr + 1 == svertices.end() ? svertices.front() : *(itr + 1);
            const FVector3 posvc = vc->getPosition();
            const FVector3 posvn = vn->getPosition();
            const FVector3 triNorm = Magnum::Math::cross(posvc - scent, posvn - scent);
            if(triNorm.isZero()) 
                continue;
            FVector3 g = (posvc - posvn) / svertices.size();
            if(vc == v) 
                g += posvn - scent;
            else if(vn == v) 
                g -= posvc - scent;
            f += Magnum::Math::cross(triNorm.normalized(), g);
        }
    }

    return 0.25 * lam * f;
}

static inline void countNeighborSurfaces(
    const Surface *s, 
    const Vertex *v, 
    const Vertex *vp, 
    const Vertex *vn, 
    const std::unordered_set<int> &targetTypes, 
    size_t &count_vp, 
    size_t &count_vn) 
{
    for(auto &sv : v->getSurfaces()) 
        if(sv->objectId() > s->objectId() && targetTypes.find(sv->typeId) != targetTypes.end()) {
            if(vp->defines(sv)) 
                ++count_vp;
            if(vn->defines(sv)) 
                ++count_vn;
        }
}

static FloatP_t Adhesion_energy_Surface(const Surface *s, const Vertex *v, const FloatP_t &lam, const std::unordered_set<int> &targetTypes) {
    Vertex *vp, *vn;
    std::tie(vp, vn) = s->neighborVertices(v);

    size_t count_vp = 0, count_vn = 0;
    countNeighborSurfaces(s, v, vp, vn, targetTypes, count_vp, count_vn);
    if(count_vp + count_vn == 0) 
        return 0;

    const FVector3 posv = v->getPosition();
    FVector3 posvp_rel = metrics::relativePosition(vp->getPosition(), posv);
    FVector3 posvn_rel = metrics::relativePosition(vn->getPosition(), posv);

    return lam * (posvp_rel.length() * count_vp + posvn_rel.length() * count_vn);
}

static FVector3 Adhesion_force_Surface(const Surface *s, const Vertex *v, const FloatP_t &lam, const std::unordered_set<int> &targetTypes) {
    Vertex *vp, *vn;
    std::tie(vp, vn) = s->neighborVertices(v);

    size_t count_vp = 0, count_vn = 0;
    countNeighborSurfaces(s, v, vp, vn, targetTypes, count_vp, count_vn);
    if(count_vp + count_vn == 0) 
        return FVector3(0);

    FVector3 force(0);

    if(count_vp > 0) {
        const FVector3 posvp_rel = metrics::relativePosition(vp->getPosition(), v->getPosition());
        if(!posvp_rel.isZero()) 
            force += posvp_rel.normalized() * count_vp;
    }
    if(count_vn > 0) {
        const FVector3 posvn_rel = metrics::relativePosition(vn->getPosition(), v->getPosition());
        if(!posvn_rel.isZero()) 
            force += posvn_rel.normalized() * count_vn;
    }

    return lam * force;
}

FloatP_t Adhesion::energy(const Surface *source, const Vertex *target) {
    auto itr = typePairs.find(source->typeId);
    if(itr == typePairs.end()) 
        return 0;
    return Adhesion_energy_Surface(source, target, lam, itr->second);
}

FVector3 Adhesion::force(const Surface *source, const Vertex *target) {
    auto itr = typePairs.find(source->typeId);
    if(itr == typePairs.end()) 
        return FVector3(0);
    return Adhesion_force_Surface(source, target, lam, itr->second);
}

FloatP_t Adhesion::energy(const Body *source, const Vertex *target) {
    auto itr = typePairs.find(source->typeId);
    if(itr == typePairs.end()) 
        return 0;
    return Adhesion_energy_Body(source, target, lam, itr->second);
}

FVector3 Adhesion::force(const Body *source, const Vertex *target) {
    auto itr = typePairs.find(source->typeId);
    if(itr == typePairs.end()) 
        return FVector3(0);
    return Adhesion_force_Body(source, target, lam, itr->second);
}

namespace TissueForge::io { 


    template <>
    HRESULT toFile(Adhesion *dataElement, const MetaData &metaData, IOElement &fileElement) { 

        TF_IOTOEASY(fileElement, metaData, "lam", dataElement->lam);
        TF_IOTOEASY(fileElement, metaData, "typePairs", dataElement->getTypePairs());

        fileElement.get()->type = "Adhesion";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Adhesion **dataElement) { 

        if(!FIO::hasImport()) 
            return tf_error(E_FAIL, "No import data available");
        else if(!TissueForge::models::vertex::io::VertexSolverFIOModule::hasImport()) 
            return tf_error(E_FAIL, "No vertex import data available");

        TissueForge::models::vertex::MeshSolver *solver = TissueForge::models::vertex::MeshSolver::get();
        if(!solver) 
            return tf_error(E_FAIL, "No vertex solver available");

        FloatP_t lam;
        TF_IOFROMEASY(fileElement, metaData, "lam", &lam);
        *dataElement = new Adhesion(lam);

        std::unordered_map<int, std::unordered_set<int> > typePairs;
        TF_IOFROMEASY(fileElement, metaData, "typePairs", &typePairs);
        for(auto &mitr : typePairs) {
            auto stype1Id_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceTypeIdMap.find(mitr.first);
            if(stype1Id_itr == TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceTypeIdMap.end()) 
                return tf_error(E_FAIL, "Could not identify type");

            for(auto &stype2Id : mitr.second) {
                auto stype2Id_itr = TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceTypeIdMap.find(stype2Id);
                if(stype2Id_itr == TissueForge::models::vertex::io::VertexSolverFIOModule::importSummary->surfaceTypeIdMap.end()) 
                    return tf_error(E_FAIL, "Could not identify type");

                if(!(*dataElement)->hasPair(stype1Id_itr->second, stype2Id_itr->second)) 
                    (*dataElement)->registerPair(stype1Id_itr->second, stype2Id_itr->second);
            }
        }

        return S_OK;
    }

};

Adhesion *Adhesion::fromString(const std::string &str) {
    return TissueForge::io::fromString<Adhesion*>(str);
}
