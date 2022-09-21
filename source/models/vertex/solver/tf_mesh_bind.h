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

#ifndef _MODELS_VERTEX_SOLVER_TF_MESH_BIND_H_
#define _MODELS_VERTEX_SOLVER_TF_MESH_BIND_H_

#include "tfMeshObj.h"
#include "tfVertex.h"
#include "tfSurface.h"
#include "tfBody.h"
#include "tfStructure.h"


namespace TissueForge::models::vertex { 


    namespace bind { 


        CPPAPI_FUNC(HRESULT) structure(StructureType *st, MeshObjActor *a);

        CPPAPI_FUNC(HRESULT) structure(Structure *s, MeshObjActor *a);

        CPPAPI_FUNC(HRESULT) body(BodyType *b, MeshObjActor *a);

        CPPAPI_FUNC(HRESULT) body(Body *b, MeshObjActor *a);

        CPPAPI_FUNC(HRESULT) surface(SurfaceType *s, MeshObjActor *a);

        CPPAPI_FUNC(HRESULT) surface(Surface *s, MeshObjActor *a);

        CPPAPI_FUNC(HRESULT) types(MeshObjType *type1, MeshObjType *type2, MeshObjTypePairActor *a);

    }

}

#endif // _MODELS_VERTEX_SOLVER_TF_MESH_BIND_H_