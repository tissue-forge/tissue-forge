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
 * @file tfMeshSolver.h
 * 
 */

#ifndef _MODELS_VERTEX_SOLVER_TFMESHSOLVER_H_
#define _MODELS_VERTEX_SOLVER_TFMESHSOLVER_H_

#include <tf_port.h>

#include "tfMesh.h"
#include "tfMeshLogger.h"

#include <tfSubEngine.h>
#include <tf_cycle.h>


namespace TissueForge::models::vertex { 


    class MeshRenderer;


    /**
     * @brief Calculate the force on a vertex
     * 
     * @param v vertex
     * @param f force
     */
    HRESULT VertexForce(const Vertex *v, FloatP_t *f);

    /**
     * @brief Calculate the force on a vertex
     * 
     * @param v vertex
     * @param f force
     */
    static HRESULT VertexForce(const Vertex *v, FVector3 &f) { return VertexForce(v, f.data()); }

    /**
     * @brief Calculate the force on a vertex
     * 
     * @param v vertex
     */
    static FVector3 VertexForce(const Vertex *v) {
        FVector3 f;
        VertexForce(v, f.data());
        return f;
    }


    /**
     * @brief Mesh solver performance timers
     */
    struct CAPI_EXPORT MeshSolverTimers {

        /**
         * @brief Solver routine sections
         */
        enum Section : unsigned int {
            FORCE=0,
            ADVANCE,
            UPDATE,
            QUALITY,
            RENDERING,
            LAST
        };

        /**
         * @brief Append time to a section
         * 
         * @param section solver section
         * @param _ticks time
         */
        HRESULT append(const Section &section, const ticks _ticks) { timers[section] += _ticks; return S_OK; }

        /**
         * @brief Get the recorded runtime of a solver section, in ms
         * 
         * @param section solver section
         * @param avg flag to return average (when true) or total time
         * @return double 
         */
        double ms(const Section &section, const bool &avg=true) const;

        /**
         * @brief Reset the timers
         */
        HRESULT reset() {
            for(size_t i = 0; i < Section::LAST; i++) timers[i] = 0;
            return S_OK;
        }

        /**
         * @brief Get a string representation of the current timer average values
         */
        std::string str() const;

    private:

        ticks timers[Section::LAST];
        
    };


    /**
     * @brief Convenience class to time mesh solver performance. 
     * 
     * Time is recorded when an instance is destroyed.
     */
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
     * A singleton solver performs all vertex model dynamics simulation at runtime. 
     */
    struct CAPI_EXPORT MeshSolver : SubEngine { 

        const char *name = "MeshSolver";

        Mesh *mesh;

        /** Performance timers */
        MeshSolverTimers timers;

        /**
         * @brief Initialize the solver
         */
        static HRESULT init();

        /**
         * @brief Get the solver singleton
         */
        static MeshSolver *get();

        /**
         * @brief Reduce internal buffers and storage
         */
        static HRESULT compact();

        /**
         * @brief Locks the engine for thread-safe engine operations
         */
        static HRESULT engineLock();
        
        /**
         * @brief Unlocks the engine for thread-safe engine operations
         */
        static HRESULT engineUnlock();

        /**
         * @brief Test whether the current mesh state needs updated
         */
        static bool isDirty();

        /** Set whether the current mesh state needs updated */
        static HRESULT setDirty(const bool &_isDirty);

        /**
         * @brief Get the mesh
         */
        static Mesh *getMesh();

        /**
         * @brief Test whether the mesh is 3D. 
         * 
         * A 3D mesh has at least one body. 
         * 
         * @return true if the mesh is 3D
         */
        static bool is3D();

        /**
         * @brief Register a body type
         * 
         * @param _type type
         */
        static HRESULT registerType(BodyType *_type);

        /**
         * @brief Register a surface type
         * 
         * @param _type type
         */
        static HRESULT registerType(SurfaceType *_type);

        /**
         * @brief Find a registered surface type by name
         * 
         * @param _name type name
         */
        static SurfaceType *findSurfaceFromName(const std::string &_name);

        /**
         * @brief Find a registered body type by name
         * 
         * @param _name type name
         */
        static BodyType *findBodyFromName(const std::string &_name);

        /**
         * @brief Get a body type by id
         * 
         * @param typeId type id
         */
        static BodyType *getBodyType(const unsigned int &typeId);

        /**
         * @brief Get a surface type by id
         * 
         * @param typeId type id
         */
        static SurfaceType *getSurfaceType(const unsigned int &typeId);

        /**
         * @brief Get the number of registered body types
         */
        static const int numBodyTypes();

        /**
         * @brief Get the number of registered surface types
         */
        static const int numSurfaceTypes();

        /**
         * @brief Get the number of vertices
         */
        static unsigned int numVertices();

        /**
         * @brief Get the number of surfaces
         */
        static unsigned int numSurfaces();

        /**
         * @brief Get the number of bodies
         */
        static unsigned int numBodies();

        /**
         * @brief Get the size of the list of vertices
         */
        static unsigned int sizeVertices();

        /**
         * @brief Get the size of the list of surfaces
         */
        static unsigned int sizeSurfaces();

        /**
         * @brief Get the size of the list of bodies
         */
        static unsigned int sizeBodies();

        /**
         * @brief Update internal data due to a change in position
         */
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

        /**
         * @brief Get the starting vertex index for each surface
         */
        static std::vector<unsigned int> getSurfaceVertexIndices();

        /**
         * @brief Start getting the starting vertex index for each surface
         */
        static HRESULT getSurfaceVertexIndicesAsyncStart();

        /**
         * @brief Finish getting the starting vertex index for each surface
         */
        static std::vector<unsigned int> getSurfaceVertexIndicesAsyncJoin();

        /**
         * @brief Get the current logger events
         */
        static std::vector<MeshLogEvent> getLog() {
            return MeshLogger::events();
        }

        /**
         * @brief Log an event
         * 
         * @param type type of event
         * @param objIDs object ids
         * @param objTypes object type ids
         * @param name name of event
         */
        static HRESULT log(const MeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MeshObjTypeLabel> &objTypes, const std::string &name="");

        friend MeshRenderer;

    private:

        FloatP_t *_forces;
        unsigned int _bufferSize;
        unsigned int _surfaceVertices;
        unsigned int _totalVertices;
        bool _isDirty;
        std::mutex _engineLock;
        std::vector<unsigned int> _surfaceVertexIndices;

        std::vector<BodyType*> _bodyTypes;
        std::vector<SurfaceType*> _surfaceTypes;

        /** Reduce internal buffers and storage */
        HRESULT _compactInst();

        /** Test whether the current mesh state needs updated */
        bool _isDirtyInst() const;

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

        /** Find a registered surface type by name */
        SurfaceType *_findSurfaceFromNameInst(const std::string &_name);

        /** Find a registered body type by name */
        BodyType *_findBodyFromNameInst(const std::string &_name);

        /** Get the body type by id */
        BodyType *_getBodyTypeInst(const unsigned int &typeId) const;

        /** Get the surface type by id */
        SurfaceType *_getSurfaceTypeInst(const unsigned int &typeId) const;

        /** Update internal data due to a change in position */
        HRESULT _positionChangedInst();

        /** Get the starting vertex index for each surface */
        std::vector<unsigned int> _getSurfaceVertexIndicesInst() const;

        /** Start getting the starting vertex index for each surface */
        HRESULT _getSurfaceVertexIndicesAsyncStartInst();

        /** Finish getting the starting vertex index for each surface */
        std::vector<unsigned int> _getSurfaceVertexIndicesAsyncJoinInst();

        /** Log an event */
        HRESULT _logInst(const MeshLogEventType &type, const std::vector<int> &objIDs, const std::vector<MeshObjTypeLabel> &objTypes, const std::string &name="");
        
    };

}

#endif // _MODELS_VERTEX_SOLVER_TFMESHSOLVER_H_