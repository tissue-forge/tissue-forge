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
#include <cycle.h>


namespace TissueForge::models::vertex { 


    class MeshRenderer;


    /**
     * @brief Calculate the force on a vertex
     * 
     * @param v vertex
     * @param f force
     */
    HRESULT VertexForce(Vertex *v, FloatP_t *f);

    /**
     * @brief Calculate the force on a vertex
     * 
     * @param v vertex
     * @param f force
     */
    static HRESULT VertexForce(Vertex *v, FVector3 &f) { return VertexForce(v, f.data()); }


    /** Mesh solver performance timers */
    struct CAPI_EXPORT MeshSolverTimers {

        /** Solver routine sections */
        enum Section : unsigned int {
            FORCE=0,
            ADVANCE,
            UPDATE,
            QUALITY,
            RENDERING,
            LAST
        };

        /** Append time to a section */
        HRESULT append(const Section &section, const ticks _ticks) { timers[section] += _ticks; return S_OK; }

        /**
         * @brief Get the recorded runtime of a solver section, in ms
         * 
         * @param section solver section
         * @param avg flag to return average (when true) or total time
         * @return double 
         */
        double ms(const Section &section, const bool &avg=true);

        /** Reset the timers */
        HRESULT reset() {
            for(size_t i = 0; i < Section::LAST; i++) timers[i] = 0;
            return S_OK;
        }

        /** Get a string representation of the current timer average values */
        std::string str();

    private:

        ticks timers[Section::LAST];
        
    };


    /** Convenience class to time mesh solver performance. Time is recorded when an instance is destroyed. */
    class CAPI_EXPORT MeshSolverTimerInstance {
        
        MeshSolverTimers::Section section;
        ticks tic;

    public:

        MeshSolverTimerInstance(const MeshSolverTimers::Section &_section);
        ~MeshSolverTimerInstance();

    };


    /**
     * @brief Vertex model mesh solver
     * 
     * A singleton solver performances all vertex model dynamics simulation at runtime. 
     */
    struct CAPI_EXPORT MeshSolver : SubEngine { 

        const char *name = "MeshSolver";

        /** Registered meshes */
        std::vector<Mesh*> meshes;

        /** Performance timers */
        MeshSolverTimers timers;

        /** Initialize the solver */
        static HRESULT init();

        /** Get the solver singleton */
        static MeshSolver *get();

        /** Reduce internal buffers and storage */
        static HRESULT compact();

        /** Locks the engine for thread-safe engine operations */
        static HRESULT engineLock();
        
        /** Unlocks the engine for thread-safe engine operations */
        static HRESULT engineUnlock();

        /** Test whether the current mesh state needs updated */
        static bool isDirty();

        /** Set whether the current mesh state needs updated */
        static HRESULT setDirty(const bool &_isDirty);

        /** Create and load a new mesh */
        static Mesh *newMesh();

        /** Load an existing mesh */
        static HRESULT loadMesh(Mesh *mesh);

        /** Unload an existing mesh */
        static HRESULT unloadMesh(Mesh *mesh);

        /** Register a body type */
        static HRESULT registerType(BodyType *_type);

        /** Register a surface type */
        static HRESULT registerType(SurfaceType *_type);

        /** Get the structure type by id */
        static StructureType *getStructureType(const unsigned int &typeId);

        /** Get the body type by id */
        static BodyType *getBodyType(const unsigned int &typeId);

        /** Get the surface type by id */
        static SurfaceType *getSurfaceType(const unsigned int &typeId);

        /** Update internal data due to a change in position */
        static HRESULT positionChanged();

        /**
         * @brief Update the solver if dirty
         * 
         * @param _force flag to force an update and ignore whether the solver is dirty
         */
        static HRESULT update(const bool &_force=false);

        HRESULT preStepStart() override;
        HRESULT preStepJoin() override;
        HRESULT postStepStart() override;
        HRESULT postStepJoin() override;

        /** Get the starting vertex index for each surface */
        static std::vector<unsigned int> getSurfaceVertexIndices();

        /** Start getting the starting vertex index for each surface */
        static HRESULT getSurfaceVertexIndicesAsyncStart();

        /** Finish getting the starting vertex index for each surface */
        static std::vector<unsigned int> getSurfaceVertexIndicesAsyncJoin();

        /** Get the current logger events */
        static std::vector<MeshLogEvent> getLog() {
            return MeshLogger::events();
        }

        /** Log an event */
        static HRESULT log(Mesh *mesh, const MeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MeshObj::Type> &objTypes, const std::string &name="");

        friend MeshRenderer;

    private:

        FloatP_t *_forces;
        unsigned int _bufferSize;
        unsigned int _surfaceVertices;
        unsigned int _totalVertices;
        bool _isDirty;
        std::mutex _engineLock;
        std::vector<unsigned int> _surfaceVertexIndices;

        std::vector<StructureType*> _structureTypes;
        std::vector<BodyType*> _bodyTypes;
        std::vector<SurfaceType*> _surfaceTypes;

        /** Reduce internal buffers and storage */
        HRESULT _compactInst();

        /** Test whether the current mesh state needs updated */
        bool _isDirtyInst();

        /** Set whether the current mesh state needs updated */
        HRESULT _setDirtyInst(const bool &_isDirty);

        /** Create and load a new mesh */
        Mesh *_newMeshInst();

        /** Load an existing mesh */
        HRESULT _loadMeshInst(Mesh *mesh);

        /** Unload an existing mesh */
        HRESULT _unloadMeshInst(Mesh *mesh);

        /** Register a body type */
        HRESULT _registerTypeInst(BodyType *_type);

        /** Register a surface type */
        HRESULT _registerTypeInst(SurfaceType *_type);

        /** Get the structure type by id */
        StructureType *_getStructureTypeInst(const unsigned int &typeId);

        /** Get the body type by id */
        BodyType *_getBodyTypeInst(const unsigned int &typeId);

        /** Get the surface type by id */
        SurfaceType *_getSurfaceTypeInst(const unsigned int &typeId);

        /** Update internal data due to a change in position */
        HRESULT _positionChangedInst();

        /** Get the starting vertex index for each surface */
        std::vector<unsigned int> _getSurfaceVertexIndicesInst();

        /** Start getting the starting vertex index for each surface */
        HRESULT _getSurfaceVertexIndicesAsyncStartInst();

        /** Finish getting the starting vertex index for each surface */
        std::vector<unsigned int> _getSurfaceVertexIndicesAsyncJoinInst();

        /** Log an event */
        HRESULT _logInst(Mesh *mesh, const MeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MeshObj::Type> &objTypes, const std::string &name="");
        
    };

}

#endif // _MODELS_VERTEX_SOLVER_TFMESHSOLVER_H_