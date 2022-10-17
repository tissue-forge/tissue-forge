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

#ifndef _MODELS_VERTEX_SOLVER_TFSTRUCTURE_H_
#define _MODELS_VERTEX_SOLVER_TFSTRUCTURE_H_

#include <tf_port.h>

#include "tf_mesh.h"


namespace TissueForge::models::vertex { 


    class Vertex;
    class Surface;
    class Body;

    struct StructureType;


    class CAPI_EXPORT Structure : public MeshObj {

        std::vector<Structure*> structures_parent;
        std::vector<Structure*> structures_child;
        std::vector<Body*> bodies;

    public:

        unsigned int typeId;

        Structure() : MeshObj() {};

        MeshObj::Type objType() { return MeshObj::Type::STRUCTURE; }

        std::vector<MeshObj*> parents();

        std::vector<MeshObj*> children() { return TissueForge::models::vertex::vectorToBase(structures_child); }

        HRESULT addChild(MeshObj *obj);

        HRESULT addParent(MeshObj *obj);

        HRESULT removeChild(MeshObj *obj);

        HRESULT removeParent(MeshObj *obj);

        HRESULT destroy();

        bool validate() { return true; }

        StructureType *type();

        std::vector<Structure*> getStructures() { return structures_parent; }

        std::vector<Body*> getBodies();

        std::vector<Surface*> getSurfaces();

        std::vector<Vertex*> getVertices();

    };


    struct CAPI_EXPORT StructureType : MeshObjType {

        MeshObj::Type objType() { return MeshObj::Type::STRUCTURE; }
        
    };

}

#endif // _MODELS_VERTEX_SOLVER_TFSTRUCTURE_H_