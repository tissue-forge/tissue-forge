/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022, 2023 T.J. Sego
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

#include "tfC_event.h"

#include "TissueForge_c_private.h"

#include <event/tfEvent.h>
#include <event/tfParticleEvent.h>
#include <event/tfTimeEvent.h>
#include <event/tfParticleTimeEvent.h>


using namespace TissueForge;


////////////////////////
// Function factories //
////////////////////////


// tfEventEventMethodHandleFcn

static tfEventEventMethodHandleFcn _EventMethod_factory_evalFcn;

HRESULT EventMethod_eval(const event::Event &e) {
    tfEventEventHandle eHandle {(void*)&e};
    
    return (*_EventMethod_factory_evalFcn)(&eHandle);
}

event::EventMethod *EventMethod_factory(tfEventEventMethodHandleFcn &fcn) {
    _EventMethod_factory_evalFcn = fcn;
    return new event::EventMethod(EventMethod_eval);
}

// tfEventParticleEventMethodHandleFcn

static tfEventParticleEventMethodHandleFcn _ParticleEventMethod_factory_evalFcn;

HRESULT ParticleEventMethod_eval(const event::ParticleEvent &e) {
    tfEventParticleEventHandle eHandle {(void*)&e};
    
    return (*_ParticleEventMethod_factory_evalFcn)(&eHandle);
}

event::ParticleEventMethod *ParticleEventMethod_factory(tfEventParticleEventMethodHandleFcn fcn) {
    _ParticleEventMethod_factory_evalFcn = fcn;
    return new event::ParticleEventMethod(ParticleEventMethod_eval);
}

// tfEventTimeEventMethodHandleFcn

static tfEventTimeEventMethodHandleFcn _TimeEventMethod_factory_evalFcn;

HRESULT TimeEventMethod_eval(const event::TimeEvent &e) {
    tfEventTimeEventHandle eHandle {(void*)&e};
    
    return (*_TimeEventMethod_factory_evalFcn)(&eHandle);
}

event::TimeEventMethod *TimeEventMethod_factory(tfEventTimeEventMethodHandleFcn fcn) {
    _TimeEventMethod_factory_evalFcn = fcn;
    return new event::TimeEventMethod(TimeEventMethod_eval);
}

// tfEventParticleTimeEventMethodHandleFcn

static tfEventParticleTimeEventMethodHandleFcn _ParticleTimeEventMethod_factory_evalFcn;

HRESULT ParticleTimeEventMethod_eval(const event::ParticleTimeEvent &e) {
    tfEventParticleTimeEventHandle eHandle {(void*)&e};
    
    return (*_ParticleTimeEventMethod_factory_evalFcn)(&eHandle);
}

event::ParticleTimeEventMethod *ParticleTimeEventMethod_factory(tfEventParticleTimeEventMethodHandleFcn fcn) {
    _ParticleTimeEventMethod_factory_evalFcn = fcn;
    return new event::ParticleTimeEventMethod(ParticleTimeEventMethod_eval);
}


//////////////////
// Module casts //
//////////////////


namespace TissueForge { 
    

    event::Event *castC(struct tfEventEventHandle *handle) {
        return castC<event::Event, tfEventEventHandle>(handle);
    }

    event::ParticleEvent *castC(struct tfEventParticleEventHandle *handle) {
        return castC<event::ParticleEvent, tfEventParticleEventHandle>(handle);
    }

    event::TimeEvent *castC(struct tfEventTimeEventHandle *handle) {
        return castC<event::TimeEvent, tfEventTimeEventHandle>(handle);
    }

    event::ParticleTimeEvent *castC(struct tfEventParticleTimeEventHandle *handle) {
        return castC<event::ParticleTimeEvent, tfEventParticleTimeEventHandle>(handle);
    }

}

#define TFC_EVENTHANDLE_GET(handle, varname) \
    event::Event *varname = TissueForge::castC<event::Event, tfEventEventHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_PARTICLEEVENTHANDLE_GET(handle, varname) \
    event::ParticleEvent *varname = TissueForge::castC<event::ParticleEvent, tfEventParticleEventHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_TIMEEVENTHANDLE_GET(handle, varname) \
    event::TimeEvent *varname = TissueForge::castC<event::TimeEvent, tfEventTimeEventHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_PARTICLETIMEEVENTHANDLE_GET(handle, varname) \
    event::ParticleTimeEvent *varname = TissueForge::castC<event::ParticleTimeEvent, tfEventParticleTimeEventHandle>(handle); \
    TFC_PTRCHECK(varname);


//////////////////////////////////////////////
// event::ParticleEventParticleSelectorEnum //
//////////////////////////////////////////////


HRESULT tfEventParticleEventParticleSelectorEnum_init(struct tfEventParticleEventParticleSelectorEnumHandle *handle) {
    handle->LARGEST = (unsigned int)event::ParticleEventParticleSelectorEnum::LARGEST; 
    handle->UNIFORM = (unsigned int)event::ParticleEventParticleSelectorEnum::UNIFORM;
    handle->DEFAULT = (unsigned int)event::ParticleEventParticleSelectorEnum::DEFAULT;
    return S_OK;
}


////////////////////////////////////
// event::TimeEventTimeSetterEnum //
////////////////////////////////////


HRESULT tfEventTimeEventTimeSetterEnum_init(struct tfEventTimeEventTimeSetterEnumHandle *handle) { 
    handle->DEFAULT = (unsigned int)event::TimeEventTimeSetterEnum::DEFAULT;
    handle->DETERMINISTIC = (unsigned int)event::TimeEventTimeSetterEnum::DETERMINISTIC;
    handle->EXPONENTIAL = (unsigned int)event::TimeEventTimeSetterEnum::EXPONENTIAL;
    return S_OK;
}


//////////////////////////////////////////////////
// event::ParticleTimeEventParticleSelectorEnum //
//////////////////////////////////////////////////


HRESULT tfEventParticleTimeEventParticleSelectorEnum_init(struct tfEventParticleTimeEventParticleSelectorEnumHandle *handle) {
    handle->LARGEST = (unsigned int)event::ParticleTimeEventParticleSelectorEnum::LARGEST; 
    handle->UNIFORM = (unsigned int)event::ParticleTimeEventParticleSelectorEnum::UNIFORM;
    handle->DEFAULT = (unsigned int)event::ParticleTimeEventParticleSelectorEnum::DEFAULT;
    return S_OK;
}


////////////////////////////////////////////
// event::ParticleTimeEventTimeSetterEnum //
////////////////////////////////////////////


HRESULT tfEventParticleTimeEventTimeSetterEnum_init(struct tfEventParticleTimeEventTimeSetterEnumHandle *handle) {
    handle->DETERMINISTIC = (unsigned int)event::ParticleTimeEventTimeSetterEnum::DETERMINISTIC;
    handle->EXPONENTIAL = (unsigned int)event::ParticleTimeEventTimeSetterEnum::EXPONENTIAL;
    handle->DEFAULT = (unsigned int)event::ParticleTimeEventTimeSetterEnum::DEFAULT;
    return S_OK;
}


//////////////////
// event::Event //
//////////////////


HRESULT tfEventEvent_getLastFired(struct tfEventEventHandle *handle, tfFloatP_t *last_fired) {
    TFC_EVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(last_fired);
    *last_fired = ev->last_fired;
    return S_OK;
}

HRESULT tfEventEvent_getTimesFired(struct tfEventEventHandle *handle, unsigned int *times_fired) {
    TFC_EVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(times_fired);
    *times_fired = ev->times_fired;
    return S_OK;
}

HRESULT tfEventEvent_remove(struct tfEventEventHandle *handle) {
    TFC_EVENTHANDLE_GET(handle, ev);
    ev->remove();
    return S_OK;
}


//////////////////////////
// event::ParticleEvent //
//////////////////////////


HRESULT tfEventParticleEvent_getLastFired(struct tfEventParticleEventHandle *handle, tfFloatP_t *last_fired) {
    TFC_PARTICLEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(last_fired);
    *last_fired = ev->last_fired;
    return S_OK;
}

HRESULT tfEventParticleEvent_getTimesFired(struct tfEventParticleEventHandle *handle, unsigned int *times_fired) {
    TFC_PARTICLEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(times_fired);
    *times_fired = ev->times_fired;
    return S_OK;
}

HRESULT tfEventParticleEvent_remove(struct tfEventParticleEventHandle *handle) {
    TFC_PARTICLEEVENTHANDLE_GET(handle, ev);
    ev->remove();
    return S_OK;
}

HRESULT tfEventParticleEvent_getTargetType(struct tfEventParticleEventHandle *handle, struct tfParticleTypeHandle *targetType) {
    TFC_PARTICLEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(targetType);
    TFC_PTRCHECK(ev->targetType);
    targetType->tfObj = (void*)ev->targetType;
    return S_OK;
}

HRESULT tfEventParticleEvent_getTargetParticle(struct tfEventParticleEventHandle *handle, struct tfParticleHandleHandle *targetParticle) {
    TFC_PARTICLEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(targetParticle);
    TFC_PTRCHECK(ev->targetParticle);
    targetParticle->tfObj = (void*)ev->targetParticle;
    return S_OK;
}


//////////////////////
// event::TimeEvent //
//////////////////////


HRESULT tfEventTimeEvent_getLastFired(struct tfEventTimeEventHandle *handle, tfFloatP_t *last_fired) {
    TFC_TIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(last_fired);
    *last_fired = ev->last_fired;
    return S_OK;
}

HRESULT tfEventTimeEvent_getTimesFired(struct tfEventTimeEventHandle *handle, unsigned int *times_fired) {
    TFC_TIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(times_fired);
    *times_fired = ev->times_fired;
    return S_OK;
}

HRESULT tfEventTimeEvent_remove(struct tfEventTimeEventHandle *handle) {
    TFC_TIMEEVENTHANDLE_GET(handle, ev);
    ev->remove();
    return S_OK;
}

HRESULT tfEventTimeEvent_getNextTime(struct tfEventTimeEventHandle *handle, tfFloatP_t *next_time) {
    TFC_TIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(next_time);
    *next_time = ev->next_time;
    return S_OK;
}

HRESULT tfEventTimeEvent_getPeriod(struct tfEventTimeEventHandle *handle, tfFloatP_t *period) {
    TFC_TIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(period);
    *period = ev->period;
    return S_OK;
}

HRESULT tfEventTimeEvent_getStartTime(struct tfEventTimeEventHandle *handle, tfFloatP_t *start_time) {
    TFC_TIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(start_time);
    *start_time = ev->start_time;
    return S_OK;
}

HRESULT tfEventTimeEvent_getEndTime(struct tfEventTimeEventHandle *handle, tfFloatP_t *end_time) {
    TFC_TIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(end_time);
    *end_time = ev->end_time;
    return S_OK;
}


//////////////////////////////
// event::ParticleTimeEvent //
//////////////////////////////


HRESULT tfEventParticleTimeEvent_getLastFired(struct tfEventParticleTimeEventHandle *handle, tfFloatP_t *last_fired) {
    TFC_PARTICLETIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(last_fired);
    *last_fired = ev->last_fired;
    return S_OK;
}

HRESULT tfEventParticleTimeEvent_getTimesFired(struct tfEventParticleTimeEventHandle *handle, unsigned int *times_fired) {
    TFC_PARTICLETIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(times_fired);
    *times_fired = ev->times_fired;
    return S_OK;
}

HRESULT tfEventParticleTimeEvent_remove(struct tfEventParticleTimeEventHandle *handle) {
    TFC_PARTICLETIMEEVENTHANDLE_GET(handle, ev);
    ev->remove();
    return S_OK;
}

HRESULT tfEventParticleTimeEvent_getNextTime(struct tfEventParticleTimeEventHandle *handle, tfFloatP_t *next_time) {
    TFC_PARTICLETIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(next_time);
    *next_time = ev->next_time;
    return S_OK;
}

HRESULT tfEventParticleTimeEvent_getPeriod(struct tfEventParticleTimeEventHandle *handle, tfFloatP_t *period) {
    TFC_PARTICLETIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(period);
    *period = ev->period;
    return S_OK;
}

HRESULT tfEventParticleTimeEvent_getStartTime(struct tfEventParticleTimeEventHandle *handle, tfFloatP_t *start_time) {
    TFC_PARTICLETIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(start_time);
    *start_time = ev->start_time;
    return S_OK;
}

HRESULT tfEventParticleTimeEvent_getEndTime(struct tfEventParticleTimeEventHandle *handle, tfFloatP_t *end_time) {
    TFC_PARTICLETIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(end_time);
    *end_time = ev->end_time;
    return S_OK;
}

HRESULT tfEventParticleTimeEvent_getTargetType(struct tfEventParticleTimeEventHandle *handle, struct tfParticleTypeHandle *targetType) {
    TFC_PARTICLETIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(targetType);
    TFC_PTRCHECK(ev->targetType);
    targetType->tfObj = (void*)ev->targetType;
    return S_OK;
}

HRESULT tfEventParticleTimeEvent_getTargetParticle(struct tfEventParticleTimeEventHandle *handle, struct tfParticleHandleHandle *targetParticle) {
    TFC_PARTICLETIMEEVENTHANDLE_GET(handle, ev);
    TFC_PTRCHECK(targetParticle);
    TFC_PTRCHECK(ev->targetParticle);
    targetParticle->tfObj = (void*)ev->targetParticle;
    return S_OK;
}


//////////////////////
// Module functions //
//////////////////////


HRESULT tfEventOnEvent(struct tfEventEventHandle *handle, tfEventEventMethodHandleFcn *invokeMethod, tfEventEventMethodHandleFcn *predicateMethod) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(invokeMethod);
    TFC_PTRCHECK(predicateMethod);
    event::Event *ev = event::onEvent(EventMethod_factory(*invokeMethod), EventMethod_factory(*predicateMethod));
    TFC_PTRCHECK(ev);
    handle->tfObj = (void*)ev;
    return S_OK;
}

HRESULT tfEventOnParticleEvent(
    struct tfEventParticleEventHandle *handle, 
    struct tfParticleTypeHandle *targetType, 
    unsigned int selectorEnum, 
    tfEventParticleEventMethodHandleFcn *invokeMethod, 
    tfEventParticleEventMethodHandleFcn *predicateMethod) 
{
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(targetType); TFC_PTRCHECK(targetType->tfObj);
    TFC_PTRCHECK(invokeMethod);
    TFC_PTRCHECK(predicateMethod);
    event::ParticleEvent *ev = event::onParticleEvent(
        (ParticleType*)targetType->tfObj, ParticleEventMethod_factory(*invokeMethod), ParticleEventMethod_factory(*predicateMethod)
    );
    TFC_PTRCHECK(ev);
    if(ev->setParticleEventParticleSelector((event::ParticleEventParticleSelectorEnum)selectorEnum) != S_OK) 
        return E_FAIL;
    handle->tfObj = (void*)ev;
    return S_OK;
}

HRESULT tfEventOnTimeEvent(
    struct tfEventTimeEventHandle *handle, 
    tfFloatP_t period, 
    tfEventTimeEventMethodHandleFcn *invokeMethod, 
    tfEventTimeEventMethodHandleFcn *predicateMethod, 
    unsigned int nextTimeSetterEnum, 
    tfFloatP_t start_time, 
    tfFloatP_t end_time) 
{
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(invokeMethod);
    TFC_PTRCHECK(predicateMethod);
    event::TimeEvent *ev = event::onTimeEvent(
        period, TimeEventMethod_factory(*invokeMethod), TimeEventMethod_factory(*predicateMethod), nextTimeSetterEnum, start_time, end_time
    );
    TFC_PTRCHECK(ev);
    handle->tfObj = (void*)ev;
    return S_OK;
}

HRESULT tfEventOnParticleTimeEvent(
    struct tfEventParticleTimeEventHandle *handle, 
    struct tfParticleTypeHandle *targetType, 
    tfFloatP_t period, 
    tfEventParticleTimeEventMethodHandleFcn *invokeMethod, 
    tfEventParticleTimeEventMethodHandleFcn *predicateMethod, 
    unsigned int nextTimeSetterEnum, 
    tfFloatP_t start_time, 
    tfFloatP_t end_time, 
    unsigned int particleSelectorEnum) 
{
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(targetType); TFC_PTRCHECK(targetType->tfObj);
    TFC_PTRCHECK(invokeMethod);
    TFC_PTRCHECK(predicateMethod);
    event::ParticleTimeEvent *ev = event::onParticleTimeEvent(
        (ParticleType*)targetType->tfObj, period, 
        ParticleTimeEventMethod_factory(*invokeMethod), ParticleTimeEventMethod_factory(*predicateMethod), 
        nextTimeSetterEnum, start_time, end_time, particleSelectorEnum
    );
    TFC_PTRCHECK(ev);
    handle->tfObj = (void*)ev;
    return S_OK;
}
