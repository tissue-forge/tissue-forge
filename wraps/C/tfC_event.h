/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022-2024 T.J. Sego
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
 * @file tfC_event.h
 * 
 */

#ifndef _WRAPS_C_TFC_EVENT_H_
#define _WRAPS_C_TFC_EVENT_H_

#include "tf_port_c.h"

#include "tfCParticle.h"

typedef HRESULT (*tfEventEventMethodHandleFcn)(struct tfEventEventHandle*);
typedef HRESULT (*tfEventParticleEventMethodHandleFcn)(struct tfEventParticleEventHandle*);
typedef HRESULT (*tfEventTimeEventMethodHandleFcn)(struct tfEventTimeEventHandle*);
typedef HRESULT (*tfEventParticleTimeEventMethodHandleFcn)(struct tfEventParticleTimeEventHandle*);

// Handles

struct CAPI_EXPORT tfEventParticleEventParticleSelectorEnumHandle {
    unsigned int LARGEST; 
    unsigned int UNIFORM;
    unsigned int DEFAULT;
};

struct CAPI_EXPORT tfEventTimeEventTimeSetterEnumHandle {
    unsigned int DEFAULT;
    unsigned int DETERMINISTIC;
    unsigned int EXPONENTIAL;
};

struct CAPI_EXPORT tfEventParticleTimeEventParticleSelectorEnumHandle {
    unsigned int LARGEST; 
    unsigned int UNIFORM;
    unsigned int DEFAULT;
};

struct CAPI_EXPORT tfEventParticleTimeEventTimeSetterEnumHandle {
    unsigned int DETERMINISTIC;
    unsigned int EXPONENTIAL;
    unsigned int DEFAULT;
};

/**
 * @brief Handle to a @ref event::Event instance
 * 
 */
struct CAPI_EXPORT tfEventEventHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref event::ParticleEvent instance
 * 
 */
struct CAPI_EXPORT tfEventParticleEventHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref event::TimeEvent instance
 * 
 */
struct CAPI_EXPORT tfEventTimeEventHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref event::ParticleTimeEvent instance
 * 
 */
struct CAPI_EXPORT tfEventParticleTimeEventHandle {
    void *tfObj;
};


//////////////////////////////////////////////
// event::ParticleEventParticleSelectorEnum //
//////////////////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleEventParticleSelectorEnum_init(struct tfEventParticleEventParticleSelectorEnumHandle *handle);


////////////////////////////////////
// event::TimeEventTimeSetterEnum //
////////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventTimeEventTimeSetterEnum_init(struct tfEventTimeEventTimeSetterEnumHandle *handle);


//////////////////////////////////////////////////
// event::ParticleTimeEventParticleSelectorEnum //
//////////////////////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEventParticleSelectorEnum_init(struct tfEventParticleTimeEventParticleSelectorEnumHandle *handle);


////////////////////////////////////////////
// event::ParticleTimeEventTimeSetterEnum //
////////////////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEventTimeSetterEnum_init(struct tfEventParticleTimeEventTimeSetterEnumHandle *handle);


//////////////////
// event::Event //
//////////////////


/**
 * @brief Get the last time an event was fired
 * 
 * @param handle populated handle
 * @param last_fired last time fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventEvent_getLastFired(struct tfEventEventHandle *handle, tfFloatP_t *last_fired);

/**
 * @brief Get the number of times an event has fired
 * 
 * @param handle populated handle
 * @param times_fired number of times fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventEvent_getTimesFired(struct tfEventEventHandle *handle, unsigned int *times_fired);

/**
 * @brief Designates event for removal
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventEvent_remove(struct tfEventEventHandle *handle);


//////////////////////////
// event::ParticleEvent //
//////////////////////////


/**
 * @brief Get the last time an event was fired
 * 
 * @param handle populated handle
 * @param last_fired last time fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleEvent_getLastFired(struct tfEventParticleEventHandle *handle, tfFloatP_t *last_fired);

/**
 * @brief Get the number of times an event has fired
 * 
 * @param handle populated handle
 * @param times_fired number of times fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleEvent_getTimesFired(struct tfEventParticleEventHandle *handle, unsigned int *times_fired);

/**
 * @brief Designates event for removal
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleEvent_remove(struct tfEventParticleEventHandle *handle);

/**
 * @brief Get the target particle type of this event
 * 
 * @param handle populated handle
 * @param targetType target particle type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleEvent_getTargetType(struct tfEventParticleEventHandle *handle, struct tfParticleTypeHandle *targetType);

/**
 * @brief Get the target particle of an event evaluation
 * 
 * @param handle populated handle
 * @param targetParticle target particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleEvent_getTargetParticle(struct tfEventParticleEventHandle *handle, struct tfParticleHandleHandle *targetParticle);


//////////////////////
// event::TimeEvent //
//////////////////////


/**
 * @brief Get the last time an event was fired
 * 
 * @param handle populated handle
 * @param last_fired last time fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventTimeEvent_getLastFired(struct tfEventTimeEventHandle *handle, tfFloatP_t *last_fired);

/**
 * @brief Get the number of times an event has fired
 * 
 * @param handle populated handle
 * @param times_fired number of times fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventTimeEvent_getTimesFired(struct tfEventTimeEventHandle *handle, unsigned int *times_fired);

/**
 * @brief Designates event for removal
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventTimeEvent_remove(struct tfEventTimeEventHandle *handle);

/**
 * @brief Get the next time of evaluation
 * 
 * @param handle populated handle
 * @param next_time next time of evaluation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventTimeEvent_getNextTime(struct tfEventTimeEventHandle *handle, tfFloatP_t *next_time);

/**
 * @brief Get the period of evaluation
 * 
 * @param handle populated handle
 * @param period period of evaluation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventTimeEvent_getPeriod(struct tfEventTimeEventHandle *handle, tfFloatP_t *period);

/**
 * @brief Get the start time of evaluations
 * 
 * @param handle populated handle
 * @param start_time start time of evaluations
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventTimeEvent_getStartTime(struct tfEventTimeEventHandle *handle, tfFloatP_t *start_time);

/**
 * @brief Get the end time of evaluations
 * 
 * @param handle populated handle
 * @param end_time end time of evaluations
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventTimeEvent_getEndTime(struct tfEventTimeEventHandle *handle, tfFloatP_t *end_time);


//////////////////////////////
// event::ParticleTimeEvent //
//////////////////////////////


/**
 * @brief Get the last time an event was fired
 * 
 * @param handle populated handle
 * @param last_fired last time fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEvent_getLastFired(struct tfEventParticleTimeEventHandle *handle, tfFloatP_t *last_fired);

/**
 * @brief Get the number of times an event has fired
 * 
 * @param handle populated handle
 * @param times_fired number of times fired
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEvent_getTimesFired(struct tfEventParticleTimeEventHandle *handle, unsigned int *times_fired);

/**
 * @brief Designates event for removal
 * 
 * @param handle populated handle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEvent_remove(struct tfEventParticleTimeEventHandle *handle);

/**
 * @brief Get the next time of evaluation
 * 
 * @param handle populated handle
 * @param next_time next time of evaluation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEvent_getNextTime(struct tfEventParticleTimeEventHandle *handle, tfFloatP_t *next_time);

/**
 * @brief Get the period of evaluation
 * 
 * @param handle populated handle
 * @param period period of evaluation
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEvent_getPeriod(struct tfEventParticleTimeEventHandle *handle, tfFloatP_t *period);

/**
 * @brief Get the start time of evaluations
 * 
 * @param handle populated handle
 * @param start_time start time of evaluations
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEvent_getStartTime(struct tfEventParticleTimeEventHandle *handle, tfFloatP_t *start_time);

/**
 * @brief Get the end time of evaluations
 * 
 * @param handle populated handle
 * @param end_time end time of evaluations
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEvent_getEndTime(struct tfEventParticleTimeEventHandle *handle, tfFloatP_t *end_time);

/**
 * @brief Get the target particle type of this event
 * 
 * @param handle populated handle
 * @param targetType target particle type
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEvent_getTargetType(struct tfEventParticleTimeEventHandle *handle, struct tfParticleTypeHandle *targetType);

/**
 * @brief Get the target particle of an event evaluation
 * 
 * @param handle populated handle
 * @param targetParticle target particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventParticleTimeEvent_getTargetParticle(struct tfEventParticleTimeEventHandle *handle, struct tfParticleHandleHandle *targetParticle);


//////////////////////
// Module functions //
//////////////////////


/**
 * @brief Creates an event using prescribed invoke and predicate functions
 * 
 * @param handle handle to populate
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventOnEvent(struct tfEventEventHandle *handle, tfEventEventMethodHandleFcn *invokeMethod, tfEventEventMethodHandleFcn *predicateMethod);

/**
 * @brief Creates a particle event using prescribed invoke and predicate functions
 * 
 * @param handle handle to populate
 * @param targetType target particle type
 * @param selectorEnum particle selector enum
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventOnParticleEvent(
    struct tfEventParticleEventHandle *handle, 
    struct tfParticleTypeHandle *targetType, 
    unsigned int selectorEnum, 
    tfEventParticleEventMethodHandleFcn *invokeMethod, 
    tfEventParticleEventMethodHandleFcn *predicateMethod
);

/**
 * @brief Creates a time-dependent event using prescribed invoke and predicate functions
 * 
 * @param handle handle to populate
 * @param period period of evaluations
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @param nextTimeSetterEnum enum selecting the function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventOnTimeEvent(
    struct tfEventTimeEventHandle *handle, 
    tfFloatP_t period, 
    tfEventTimeEventMethodHandleFcn *invokeMethod, 
    tfEventTimeEventMethodHandleFcn *predicateMethod, 
    unsigned int nextTimeSetterEnum, 
    tfFloatP_t start_time, 
    tfFloatP_t end_time
);

/**
 * @brief Creates a time-dependent particle event using prescribed invoke and predicate functions
 * 
 * @param handle handle to populate
 * @param targetType target particle type
 * @param period period of evaluations
 * @param invokeMethod an invoke function; evaluated when an event occurs
 * @param predicateMethod a predicate function; evaluated to determine if an event occurs
 * @param nextTimeSetterEnum enum of function that sets the next evaulation time
 * @param start_time time at which evaluations begin
 * @param end_time time at which evaluations end
 * @param particleSelectorEnum enum of function that selects the next particle
 * @return S_OK on success 
 */
CAPI_FUNC(HRESULT) tfEventOnParticleTimeEvent(
    struct tfEventParticleTimeEventHandle *handle, 
    struct tfParticleTypeHandle *targetType, 
    tfFloatP_t period, 
    tfEventParticleTimeEventMethodHandleFcn *invokeMethod, 
    tfEventParticleTimeEventMethodHandleFcn *predicateMethod, 
    unsigned int nextTimeSetterEnum, 
    tfFloatP_t start_time, 
    tfFloatP_t end_time, 
    unsigned int particleSelectorEnum
);

#endif // _WRAPS_C_TFC_EVENT_H_