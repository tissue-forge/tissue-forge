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

/**
 * @file tf_mesh_bind.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TF_MESH_BIND_H_
#define _MODELS_VERTEX_SOLVER_TF_MESH_BIND_H_

#include "tfMeshObj.h"
#include "tfVertex.h"
#include "tfSurface.h"
#include "tfBody.h"


namespace TissueForge::models::vertex { 


    namespace bind { 


        /**
         * @brief Bind an actor to a body type
         * 
         * @param a actor
         * @param b body type
         */
        CPPAPI_FUNC(HRESULT) body(MeshObjActor *a, BodyType *b);

        /**
         * @brief Bind an actor to a body
         * 
         * @param a actor
         * @param h body
         */
        CPPAPI_FUNC(HRESULT) body(MeshObjActor *a, const BodyHandle &h);

        /**
         * @brief Bind an actor to a surface type
         * 
         * @param a actor
         * @param s surface type
         */
        CPPAPI_FUNC(HRESULT) surface(MeshObjActor *a, SurfaceType *s);

        /**
         * @brief Bind an actor to a surface
         * 
         * @param a actor
         * @param h surface
         */
        CPPAPI_FUNC(HRESULT) surface(MeshObjActor *a, const SurfaceHandle &h);

        /**
         * @brief Bind an actor to a pair of object types
         * 
         * @param a actor
         * @param type1 first object type
         * @param type2 second object type
         */
        CPPAPI_FUNC(HRESULT) types(MeshObjTypePairActor *a, MeshObjType *type1, MeshObjType *type2);

    }

}

#endif // _MODELS_VERTEX_SOLVER_TF_MESH_BIND_H_