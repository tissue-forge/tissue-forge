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

#include "tfMeshObj.h"

#include "tfMeshSolver.h"

#include "actors/tfAdhesion.h"
#include "actors/tfBodyForce.h"
#include "actors/tfConvexPolygonConstraint.h"
#include "actors/tfEdgeTension.h"
#include "actors/tfFlatSurfaceConstraint.h"
#include "actors/tfNormalStress.h"
#include "actors/tfSurfaceAreaConstraint.h"
#include "actors/tfSurfaceTraction.h"
#include "actors/tfVolumeConstraint.h"

#include <tfError.h>
#include <io/tfFIO.h>

using namespace TissueForge;
using namespace TissueForge::models::vertex;


MeshObj::MeshObj() : 
    mesh{NULL}, 
    objId{-1}
{}

bool MeshObj::in(const MeshObj *obj) const {
    if(!obj || objType() > obj->objType()) 
        return false;

    for(auto &p : obj->parents()) 
        if(p == this || in(p)) 
            return true;

    return false;
}

bool MeshObj::has(const MeshObj *obj) const {
    return obj && obj->in(this);
}

HRESULT MeshObjTypePairActor::registerPair(MeshObjType *type1, MeshObjType *type2) {
    if(type1->id < 0 || type2->id < 0) {
        tf_error(E_FAIL, "Object type not registered");
        return E_FAIL;
    }

    auto &v1 = typePairs[type1->id];
    v1.insert(type2->id);

    auto &v2 = typePairs[type2->id];
    v2.insert(type1->id);

    return S_OK;
}

bool MeshObjTypePairActor::hasPair(MeshObjType *type1, MeshObjType *type2) {
    auto itr1 = typePairs.find(type1->id);
    if(itr1 != typePairs.end()) 
        return itr1->second.find(type2->id) != itr1->second.end();
    else 
        return false;
}

namespace TissueForge::io {


template <typename T> 
bool MeshObjActor_isString(const IOElement &fileElement, const MetaData &metaData) {
    auto itr = fileElement.children.find("name");
    if(itr == fileElement.children.end()) 
        return false;
    std::string name;
    if(fromFile<std::string>(*itr->second, metaData, &name) != S_OK) 
        return false;
    return strcmp(name.c_str(), T::actorName().c_str()) == 0;
}

#define MESHOBJACTOR_TOFILECAST(T, dataElement, metaData, fileElement) toFile<T>((T*)dataElement, metaData, fileElement);
#define MESHOBJACTOR_CONDTOFILECASTRET(T, dataElement, metaData, fileElement) { \
    if(strcmp(dataElement->name().c_str(), T::actorName().c_str()) == 0) {      \
        return MESHOBJACTOR_TOFILECAST(T, dataElement, metaData, fileElement);  \
    }}
#define MESHOBJACTOR_CONDFROMFILECASTRET(T, dataElement, metaData, fileElement) {   \
    if(MeshObjActor_isString<T>(fileElement, metaData)) {                           \
            T **dataElement_c = (T**)dataElement;                                   \
            return fromFile(fileElement, metaData, dataElement_c);                  \
    }                                                                               \
}


template <>
HRESULT toFile(TissueForge::models::vertex::MeshObjActor *dataElement, const MetaData &metaData, IOElement *fileElement) {
    IOElement *nameElement = new IOElement();
    fileElement->children["name"] = nameElement;
    toFile(dataElement->name(), metaData, nameElement);
    fileElement->type = "MeshObjActor";

    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::Adhesion, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::BodyForce, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::ConvexPolygonConstraint, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::EdgeTension, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::FlatSurfaceConstraint, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::NormalStress, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::SurfaceAreaConstraint, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::SurfaceTraction, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::VolumeConstraint, dataElement, metaData, fileElement);

    return S_OK;
}

template <>
HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::models::vertex::MeshObjActor **dataElement) {

    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::Adhesion, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::BodyForce, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::ConvexPolygonConstraint, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::EdgeTension, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::FlatSurfaceConstraint, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::NormalStress, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::SurfaceAreaConstraint, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::SurfaceTraction, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::VolumeConstraint, dataElement, metaData, fileElement);

    return tf_error(E_FAIL, "Could not identify actor");
}


};

std::string TissueForge::models::vertex::MeshObjActor::toString() {
    TissueForge::io::IOElement *el = new TissueForge::io::IOElement();
    std::string result;
    if(TissueForge::io::toFile(this, TissueForge::io::MetaData(), el) != S_OK) result = "";
    else result = TissueForge::io::toStr(el);
    delete el;
    return result;
}
