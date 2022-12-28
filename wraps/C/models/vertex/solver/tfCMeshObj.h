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

#ifndef _WRAPS_C_VERTEX_SOLVER_TFCMESHOBJ_H_
#define _WRAPS_C_VERTEX_SOLVER_TFCMESHOBJ_H_

#include <tf_port_c.h>

#include "tfCVertex.h"
#include "tfCSurface.h"
#include "tfCBody.h"


// Handles


/**
 * @brief Handle to a @ref models::vertex::MeshObjTypeLabel instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverMeshObjTypeLabelHandle {
    unsigned int NONE;
    unsigned int VERTEX;
    unsigned int SURFACE;
    unsigned int BODY;
};

/**
 * @brief Handle to a @ref models::vertex::MeshObjActor instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverMeshObjActorHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref models::vertex::MeshObjTypePairActor instance
 * 
 */
struct CAPI_EXPORT tfVertexSolverMeshObjTypePairActorHandle {
    void *tfObj;
};


//////////////////////
// MeshObjTypeLabel //
//////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshObjTypeLabel_init(struct tfVertexSolverMeshObjTypeLabelHandle *handle);


//////////////////
// MeshObjActor //
//////////////////


/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshObjActor_destroy(struct tfVertexSolverMeshObjActorHandle *handle);

/**
 * @brief Name of the actor
 * 
 * @param handle populated handle
 * @param str name
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshObjActor_getName(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    char **str, 
    unsigned int *numChars
);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str JSON string
 * @param numChars number of chars
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshObjActor_toString(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    char **str, 
    unsigned int *numChars
);

/**
 * @brief Calculate the energy of a source object acting on a target object
 * 
 * @param handle populated handle
 * @param source source object
 * @param target target object
 * @param result energy 
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshObjActor_getEnergySurface(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *source, 
    struct tfVertexSolverVertexHandleHandle *target, 
    tfFloatP_t *result
);

/**
 * @brief Calculate the force that a source object exerts on a target object
 * 
 * @param handle populated handle
 * @param source source object
 * @param target target object
 * @param result force
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshObjActor_getForceSurface(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverSurfaceHandleHandle *source, 
    struct tfVertexSolverVertexHandleHandle *target, 
    tfFloatP_t **result
);

/**
 * @brief Calculate the energy of a source object acting on a target object
 * 
 * @param handle populated handle
 * @param source source object
 * @param target target object
 * @param result energy 
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshObjActor_getEnergyBody(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *source, 
    struct tfVertexSolverVertexHandleHandle *target, 
    tfFloatP_t *result
);

/**
 * @brief Calculate the force that a source object exerts on a target object
 * 
 * @param handle populated handle
 * @param source source object
 * @param target target object
 * @param result force
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshObjActor_getForceBody(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverBodyHandleHandle *source, 
    struct tfVertexSolverVertexHandleHandle *target, 
    tfFloatP_t **result
);


//////////////////////////
// MeshObjTypePairActor //
//////////////////////////


/**
 * @brief Cast to a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshObjTypePairActor_toBase(
    struct tfVertexSolverMeshObjTypePairActorHandle *handle, 
    struct tfVertexSolverMeshObjActorHandle *result
);

/**
 * @brief Cast from a base actor instance
 * 
 * @param handle populated handle
 * @param result result of cast
 */
CAPI_FUNC(HRESULT) tfVertexSolverMeshObjTypePairActor_fromBase(
    struct tfVertexSolverMeshObjActorHandle *handle, 
    struct tfVertexSolverMeshObjTypePairActorHandle *result
);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Get the actors with a given name from a surface
 * 
 * @param handle populated handle
 * @param actorName name of actor
 * @param actors actors
 * @param numActors number of actors
 */
CAPI_FUNC(HRESULT) tfVertexSolver_getActorsFromSurfaceByName(
    struct tfVertexSolverSurfaceHandleHandle *handle, 
    const char *actorName, 
    struct tfVertexSolverMeshObjActorHandle **actors, 
    unsigned int *numActors
);

/**
 * @brief Get the actors with a given name from a body
 * 
 * @param handle populated handle
 * @param actorName name of actor
 * @param actors actors
 * @param numActors number of actors
 */
CAPI_FUNC(HRESULT) tfVertexSolver_getActorsFromBodyByName(
    struct tfVertexSolverBodyHandleHandle *handle, 
    const char *actorName, 
    struct tfVertexSolverMeshObjActorHandle **actors, 
    unsigned int *numActors
);

/**
 * @brief Get the actors with a given name from a surface type
 * 
 * @param handle populated handle
 * @param actorName name of actor
 * @param actors actors
 * @param numActors number of actors
 */
CAPI_FUNC(HRESULT) tfVertexSolver_getActorsFromSurfaceTypeByName(
    struct tfVertexSolverSurfaceTypeHandle *handle, 
    const char *actorName, 
    struct tfVertexSolverMeshObjActorHandle **actors, 
    unsigned int *numActors
);

/**
 * @brief Get the actors with a given name from a body type
 * 
 * @param handle populated handle
 * @param actorName name of actor
 * @param actors actors
 * @param numActors number of actors
 */
CAPI_FUNC(HRESULT) tfVertexSolver_getActorsFromBodyTypeByName(
    struct tfVertexSolverBodyTypeHandle *handle, 
    const char *actorName, 
    struct tfVertexSolverMeshObjActorHandle **actors, 
    unsigned int *numActors
);

#endif // _WRAPS_C_VERTEX_SOLVER_TFCMESHOBJ_H_