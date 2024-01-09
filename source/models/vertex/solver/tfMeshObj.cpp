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

#include "tfMeshObj.h"

#include "tfMeshSolver.h"

#include "actors/tf_actors.h"

#include <tfError.h>
#include <io/tfFIO.h>

using namespace TissueForge;
using namespace TissueForge::models::vertex;


HRESULT MeshObjTypePairActor::registerPair(const int &type1, const int &type2) {
    if(type1 < 0 || type2 < 0) {
        tf_error(E_FAIL, "Object type not registered");
        return E_FAIL;
    }

    auto &v1 = typePairs[type1];
    v1.insert(type2);

    auto &v2 = typePairs[type2];
    v2.insert(type1);

    return S_OK;
}

HRESULT MeshObjTypePairActor::registerPair(MeshObjType *type1, MeshObjType *type2) {
    return registerPair(type1->id, type2->id);
}

bool MeshObjTypePairActor::hasPair(const int &type1, const int &type2) {
    auto itr1 = typePairs.find(type1);
    if(itr1 != typePairs.end()) 
        return itr1->second.find(type2) != itr1->second.end();
    else 
        return false;
}

bool MeshObjTypePairActor::hasPair(MeshObjType *type1, MeshObjType *type2) {
    return hasPair(type1->id, type2->id);
}

namespace TissueForge::io {


template <typename T> 
bool MeshObjActor_isString(const IOElement &fileElement, const MetaData &metaData) {
    IOChildMap fec = IOElement::children(fileElement);
    auto itr = fec.find("name");
    if(itr == fec.end()) 
        return false;
    std::string name;
    if(fromFile<std::string>(itr->second, metaData, &name) != S_OK) 
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
HRESULT toFile(TissueForge::models::vertex::MeshObjActor *dataElement, const MetaData &metaData, IOElement &fileElement) {
    IOElement nameElement = IOElement::create();
    TF_IOTOEASY(fileElement, metaData, "name", dataElement->name());
    fileElement.get()->type = "MeshObjActor";

    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::Adhesion, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::BodyForce, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::ConvexPolygonConstraint, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::EdgeTension, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::FlatSurfaceConstraint, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::NormalStress, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDTOFILECASTRET(TissueForge::models::vertex::PerimeterConstraint, dataElement, metaData, fileElement);
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
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::PerimeterConstraint, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::SurfaceAreaConstraint, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::SurfaceTraction, dataElement, metaData, fileElement);
    MESHOBJACTOR_CONDFROMFILECASTRET(TissueForge::models::vertex::VolumeConstraint, dataElement, metaData, fileElement);

    return tf_error(E_FAIL, "Could not identify actor");
}


};

std::string TissueForge::models::vertex::MeshObjActor::toString() {
    TissueForge::io::IOElement el = TissueForge::io::IOElement::create();
    std::string result;
    if(TissueForge::io::toFile(this, TissueForge::io::MetaData(), el) != S_OK) result = "";
    else result = TissueForge::io::toStr(el);
    return result;
}
