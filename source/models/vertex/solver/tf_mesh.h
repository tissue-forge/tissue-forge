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

#ifndef _MODELS_VERTEX_SOLVER_TF_MESH_H_
#define _MODELS_VERTEX_SOLVER_TF_MESH_H_

#include "tfMeshObj.h"


namespace TissueForge::models::vertex {


    /** Check whether an object is an instance of an object type */
    TF_ALWAYS_INLINE bool check(MeshObj *obj, const MeshObj::Type &typeEnum) {
        return obj->objType() == typeEnum;
    }

    /** Convert a vector of derived mesh objects to a vector of base mesh objects */
    template<typename T>
    std::vector<MeshObj*> vectorToBase(const std::vector<T*> &implVec) {
        return std::vector<MeshObj*>(implVec.begin(), implVec.end());
    }

    /** Convert a vector of base mesh objects to a vector of derived mesh objects */
    template<typename T> 
    std::vector<T*> vectorToDerived(const std::vector<MeshObj*> &baseVec) {
        std::vector<T*> result(baseVec.size(), 0);
        for(unsigned int i = 0; i < baseVec.size(); i++) 
            result[i] = dynamic_cast<T*>(baseVec[i]);
        return result;
    }

}

#endif // _MODELS_VERTEX_SOLVER_TF_MESH_H_