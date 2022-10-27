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

#include "tf_mesh_bind.h"


using namespace TissueForge::models::vertex;


namespace TissueForge::models::vertex::bind { 


    HRESULT structure(MeshObjActor *a, StructureType *s) {
        s->actors.push_back(a);
        return S_OK;
    }

    HRESULT structure(MeshObjActor *a, Structure *s) {
        s->actors.push_back(a);
        return S_OK;
    }

    HRESULT body(MeshObjActor *a, BodyType *b) {
        b->actors.push_back(a);
        return S_OK;
    }

    HRESULT body(MeshObjActor *a, Body *b) {
        b->actors.push_back(a);
        return S_OK;
    }

    HRESULT surface(MeshObjActor *a, SurfaceType *s) { 
        s->actors.push_back(a);
        return S_OK;
    }

    HRESULT surface(MeshObjActor *a, Surface *s) { 
        s->actors.push_back(a);
        return S_OK;
    }

    HRESULT types(MeshObjTypePairActor *a, MeshObjType *type1, MeshObjType *type2) {
        if(a->registerPair(type1, type2) != S_OK) 
            return E_FAIL;
        type1->actors.push_back(a);
        if(type1->id != type2->id) 
            type2->actors.push_back(a);
        return S_OK;
    }

}