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

#ifndef _MODELS_VERTEX_SOLVER_TFMESHSOLVER_H_
#define _MODELS_VERTEX_SOLVER_TFMESHSOLVER_H_

#include <tf_port.h>

#include "tfMesh.h"
#include "tfMeshLogger.h"

#include <tfSubEngine.h>


namespace TissueForge::models::vertex { 


    class MeshRenderer;


    HRESULT VertexForce(Vertex *v, FloatP_t *f);


    struct CAPI_EXPORT MeshSolver : SubEngine { 

        const char *name = "MeshSolver";

        std::vector<Mesh*> meshes;

        static HRESULT init();
        static MeshSolver *get();
        HRESULT compact();

        /** Locks the engine for thread-safe engine operations */
        static HRESULT engineLock();
        
        /** Unlocks the engine for thread-safe engine operations */
        static HRESULT engineUnlock();

        bool isDirty();
        HRESULT setDirty(const bool &_isDirty);

        Mesh *newMesh();
        HRESULT loadMesh(Mesh *mesh);
        HRESULT unloadMesh(Mesh *mesh);

        HRESULT registerType(BodyType *_type);
        HRESULT registerType(SurfaceType *_type);

        StructureType *getStructureType(const unsigned int &typeId);
        BodyType *getBodyType(const unsigned int &typeId);
        SurfaceType *getSurfaceType(const unsigned int &typeId);

        HRESULT positionChanged();
        HRESULT update(const bool &_force=false);

        HRESULT preStepStart();
        HRESULT preStepJoin();
        HRESULT postStepStart();
        HRESULT postStepJoin();

        std::vector<MeshLogEvent> getLog() {
            return MeshLogger::events();
        }
        HRESULT log(Mesh *mesh, const MeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MeshObj::Type> &objTypes, const std::string &name="");

        friend MeshRenderer;

    private:

        float *_forces;
        unsigned int _bufferSize;
        unsigned int _surfaceVertices;
        unsigned int _totalVertices;
        bool _isDirty;
        std::mutex _engineLock;

        std::vector<StructureType*> _structureTypes;
        std::vector<BodyType*> _bodyTypes;
        std::vector<SurfaceType*> _surfaceTypes;
        
    };

}

#endif // _MODELS_VERTEX_SOLVER_TFMESHSOLVER_H_