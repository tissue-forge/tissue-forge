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

#include "tfConvexPolygonConstraint.h"

#include <models/vertex/solver/tfSurface.h>
#include <models/vertex/solver/tfVertex.h>

#include <tfEngine.h>
#include <io/tfFIO.h>


using namespace TissueForge;
using namespace TissueForge::models::vertex;


static inline bool ConvexPolygonConstraint_acts(const Vertex *vc, const Surface *s, FVector3 &rel_c2ab) {
    std::vector<Vertex*> vertices = s->getVertices();
    if(vertices.size() <= 3) 
        return false;

    Vertex *va, *vb;
    std::tie(va, vb) = s->neighborVertices(vc);
    const FVector3 posva = va->getPosition();
    const FVector3 posvb = vb->getPosition();

    // Perpindicular vector from vertex to line connecting neighbors should point 
    //  in the opposite direction as the vector from the vertex to the centroid of all other vertices
    // rel_c2ab = posvc.lineShortestDisplacementTo(posva, posvb);
    // const FVector3 rel_cent2ab = cent_loo.lineShortestDisplacementTo(posva, posvb);
    FVector3 lineDir = posvb - posva;
    if(lineDir.isZero()) 
        return false;
    lineDir = lineDir.normalized();

    const FVector3 posvc = vc->getPosition();
    const FloatP_t numVerts = vertices.size();
    const FVector3 cent_loo = (s->getCentroid() * numVerts - posvc) / (numVerts - 1.0);
    rel_c2ab = posva + (posvc - posva).dot(lineDir) * lineDir - posvc;
    const FVector3 rel_cent2ab = posva + (cent_loo - posva).dot(lineDir) * lineDir - cent_loo;
    
    return rel_c2ab.dot(rel_cent2ab) > 0;
}

FloatP_t ConvexPolygonConstraint::energy(const Surface *source, const Vertex *target) {
    FloatP_t e;
    FVector3 rel_c2ab;
    if(ConvexPolygonConstraint_acts(target, source, rel_c2ab)) 
        e += target->getCachedParticleMass() / _Engine.dt * lam / 2.0 * rel_c2ab.dot();

    return e;
}

FVector3 ConvexPolygonConstraint::force(const Surface *source, const Vertex *target) {
    FVector3 force;
    FVector3 rel_c2ab;
    if(ConvexPolygonConstraint_acts(target, source, rel_c2ab)) {
        force += rel_c2ab * target->getCachedParticleMass() / _Engine.dt * lam;
    }

    return force;
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
    HRESULT toFile(ConvexPolygonConstraint *dataElement, const MetaData &metaData, IOElement *fileElement) { 

        IOElement *fe;

        TF_ACTORIOTOEASY(fe, "lam", dataElement->lam);

        fileElement->type = "ConvexPolygonConstraint";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, ConvexPolygonConstraint **dataElement) { 

        IOChildMap::const_iterator feItr;

        FloatP_t lam;
        TF_ACTORIOFROMEASY(feItr, fileElement.children, metaData, "lam", &lam);
        *dataElement = new ConvexPolygonConstraint(lam);

        return S_OK;
    }

};

ConvexPolygonConstraint *ConvexPolygonConstraint::fromString(const std::string &str) {
    return TissueForge::io::fromString<ConvexPolygonConstraint*>(str);
}
