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
#include <tfError.h>

using namespace TissueForge;
using namespace TissueForge::models::vertex;


MeshObj::MeshObj() : 
    mesh{NULL}, 
    objId{-1}
{}

bool MeshObj::in(MeshObj *obj) {
    if(!obj || objType() > obj->objType()) 
        return false;

    for(auto &p : obj->parents()) 
        if(p == this || in(p)) 
            return true;

    return false;
}

bool MeshObj::has(MeshObj *obj) {
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
