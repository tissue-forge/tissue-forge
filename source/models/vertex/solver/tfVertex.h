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

#ifndef _MODELS_VERTEX_SOLVER_TFVERTEX_H_
#define _MODELS_VERTEX_SOLVER_TFVERTEX_H_

#include <tf_port.h>

#include <tfParticle.h>
#include <rendering/tfStyle.h>

#include "tfMeshObj.h"

#include <io/tfThreeDFVertexData.h>

#include <vector>


namespace TissueForge::models::vertex { 


    class Surface;
    class Body;
    class Structure;
    class Mesh;


    struct CAPI_EXPORT MeshParticleType : ParticleType { 

        MeshParticleType() : ParticleType(true) {
            std::memcpy(this->name, "MeshParticleType", sizeof("MeshParticleType"));
            style->setVisible(false);
            registerType();
        };

    };

    CAPI_FUNC(MeshParticleType*) MeshParticleType_get();


    /**
     * @brief The mesh vertex is a volume of a mesh centered at a point in a space.
     * 
     */
    class CAPI_EXPORT Vertex : public MeshObj {

        /** Particle id. -1 if not assigned */
        int pid;

        /** Connected surfaces */
        std::vector<Surface*> surfaces;

    public:

        Vertex();
        Vertex(const unsigned int &_pid);
        Vertex(const FVector3 &position);
        Vertex(io::ThreeDFVertexData *vdata);

        MeshObj::Type objType() { return MeshObj::Type::VERTEX; }

        std::vector<MeshObj*> parents() { return std::vector<MeshObj*>(); }

        std::vector<MeshObj*> children();

        HRESULT addChild(MeshObj *obj);

        HRESULT addParent(MeshObj *obj) { return E_FAIL; }

        HRESULT removeChild(MeshObj *obj);

        HRESULT removeParent(MeshObj *obj) { return E_FAIL; }

        HRESULT add(Surface *s);
        HRESULT insert(Surface *s, const int &idx);
        HRESULT insert(Surface *s, Surface *before);
        HRESULT remove(Surface *s);
        HRESULT replace(Surface *toInsert, const int &idx);
        HRESULT replace(Surface *toInsert, Surface *toRemove);

        HRESULT destroy();

        bool validate() { return true; }

        std::vector<Structure*> getStructures();

        std::vector<Body*> getBodies();

        std::vector<Surface*> getSurfaces() { return surfaces; }

        Surface *findSurface(const FVector3 &dir);
        Body *findBody(const FVector3 &dir);

        std::vector<Vertex*> neighborVertices();

        std::vector<Surface*> sharedSurfaces(Vertex *other);

        FloatP_t getVolume();
        FloatP_t getMass();

        HRESULT positionChanged();
        HRESULT updateProperties();

        ParticleHandle *particle();

        FVector3 getPosition();

        HRESULT setPosition(const FVector3 &pos);


        friend Surface;
        friend Body;
        friend Mesh;

    };

}

#endif // _MODELS_VERTEX_SOLVER_TFVERTEX_H_