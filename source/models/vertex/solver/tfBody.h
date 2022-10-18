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

#ifndef _MODELS_VERTEX_SOLVER_TFBODY_H_
#define _MODELS_VERTEX_SOLVER_TFBODY_H_

#include <tf_port.h>

#include <state/tfStateVector.h>

#include "tf_mesh.h"

#include <io/tfThreeDFMeshData.h>


namespace TissueForge::models::vertex { 


    class Vertex;
    class Surface;
    class Structure;
    class Mesh;

    struct BodyType;
    struct SurfaceType;

    /**
     * @brief The mesh body is a volume-enclosing object of mesh surfaces. 
     * 
     * The mesh body consists of at least four mesh surfaces. 
     * 
     * The mesh body can have a state vector, which represents a uniform amount of substance 
     * enclosed in the volume of the body. 
     * 
     */
    class CAPI_EXPORT Body : public MeshObj { 

        std::vector<Surface*> surfaces;

        std::vector<Structure*> structures;

        /** current centroid */
        FVector3 centroid;

        /** current surface area */
        FloatP_t area;

        /** current volume */
        FloatP_t volume;

        /** mass density */
        FloatP_t density;

        void _updateInternal();

    public:

        unsigned int typeId;

        state::StateVector *species;

        Body();

        /** Construct a body from a set of surfaces */
        Body(std::vector<Surface*> _surfaces);

        /** Construct a body from a mesh */
        Body(io::ThreeDFMeshData *ioMesh);

        MeshObj::Type objType() { return MeshObj::Type::BODY; }

        std::vector<MeshObj*> parents();

        std::vector<MeshObj*> children();

        HRESULT addChild(MeshObj *obj);

        HRESULT addParent(MeshObj *obj);

        HRESULT removeChild(MeshObj *obj);

        HRESULT removeParent(MeshObj *obj);

        HRESULT add(Surface *s);
        HRESULT remove(Surface *s);
        HRESULT replace(Surface *toInsert, Surface *toRemove);

        HRESULT add(Structure *s);
        HRESULT remove(Structure *s);
        HRESULT replace(Structure *toInsert, Structure *toRemove);

        HRESULT destroy();

        bool validate();

        HRESULT positionChanged();

        BodyType *type();

        HRESULT become(BodyType *btype);

        std::vector<Structure*> getStructures();
        std::vector<Surface*> getSurfaces() { return surfaces; }
        std::vector<Vertex*> getVertices();

        Vertex *findVertex(const FVector3 &dir);
        Surface *findSurface(const FVector3 &dir);

        std::vector<Body*> neighborBodies();
        std::vector<Surface*> neighborSurfaces(Surface *s);

        FloatP_t getDensity() const { return density; }
        void setDensity(const FloatP_t &_density) { density = _density; }

        FVector3 getCentroid() const { return centroid; }
        FVector3 getVelocity();
        FloatP_t getArea() const { return area; }
        FloatP_t getVolume() const { return volume; }
        FloatP_t getMass() const { return volume * density; }

        FloatP_t getVertexArea(Vertex *v);
        FloatP_t getVertexVolume(Vertex *v);
        FloatP_t getVertexMass(Vertex *v) { return getVertexVolume(v) * density; }

        std::vector<Surface*> findInterface(Body *b);
        FloatP_t contactArea(Body *other);

        
        friend Mesh;
        friend BodyType;

    };


    struct CAPI_EXPORT BodyType : MeshObjType {

        FloatP_t density;

        MeshObj::Type objType() { return MeshObj::Type::BODY; }

        /** Construct a body of this type from a set of surfaces */
        Body *operator() (std::vector<Surface*> surfaces);

        /** Construct a body of this type from a mesh */
        Body *operator() (io::ThreeDFMeshData* ioMesh, SurfaceType *stype);
    };

}

#endif // _MODELS_VERTEX_SOLVER_TFBODY_H_