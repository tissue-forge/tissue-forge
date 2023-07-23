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

#include "tfCSpeciesReaction.h"

#include "TissueForge_c_private.h"

#include <state/tfSpeciesReactionDef.h>
#include <state/tfSpeciesReaction.h>
#include <state/tfSpeciesReactions.h>


using namespace TissueForge;


namespace TissueForge { 


    state::SpeciesReactionDef *castC(struct tfStateSpeciesReactionDefHandle *handle) {
        return castC<state::SpeciesReactionDef, tfStateSpeciesReactionDefHandle>(handle);
    }

    state::SpeciesReaction *castC(struct tfStateSpeciesReactionHandle *handle) {
        return castC<state::SpeciesReaction, tfStateSpeciesReactionHandle>(handle);
    }

    state::SpeciesReactions *castC(struct tfStateSpeciesReactionsHandle *handle) {
        return castC<state::SpeciesReactions, tfStateSpeciesReactionsHandle>(handle);
    }

}

#define TFC_SPECIESREACTIONDEF_GET(handle, varname) \
    state::SpeciesReactionDef *varname = TissueForge::castC<state::SpeciesReactionDef, tfStateSpeciesReactionDefHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_SPECIESREACTION_GET(handle, varname) \
    state::SpeciesReaction *varname = TissueForge::castC<state::SpeciesReaction, tfStateSpeciesReactionHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_SPECIESREACTIONS_GET(handle, varname) \
    state::SpeciesReactions *varname = TissueForge::castC<state::SpeciesReactions, tfStateSpeciesReactionsHandle>(handle); \
    TFC_PTRCHECK(varname);


///////////////////////////////////////
// state::SpeciesReactionIntegrators //
///////////////////////////////////////


HRESULT tfStateSpeciesReactionIntegratorsEnum_init(struct tfStateSpeciesReactionIntegratorsEnum* handle) {
    TFC_PTRCHECK(handle);
    handle->SPECIESREACTIONINT_NONE = state::SpeciesReactionIntegrators::SPECIESREACTIONINT_NONE;
    handle->SPECIESREACTIONINT_CVODE = state::SpeciesReactionIntegrators::SPECIESREACTIONINT_CVODE;
    handle->SPECIESREACTIONINT_EULER = state::SpeciesReactionIntegrators::SPECIESREACTIONINT_EULER;
    handle->SPECIESREACTIONINT_GILLESPIE = state::SpeciesReactionIntegrators::SPECIESREACTIONINT_GILLESPIE;
    handle->SPECIESREACTIONINT_NLEQ1 = state::SpeciesReactionIntegrators::SPECIESREACTIONINT_NLEQ1;
    handle->SPECIESREACTIONINT_NLEQ2 = state::SpeciesReactionIntegrators::SPECIESREACTIONINT_NLEQ2;
    handle->SPECIESREACTIONINT_RK4 = state::SpeciesReactionIntegrators::SPECIESREACTIONINT_RK4;
    handle->SPECIESREACTIONINT_RK45 = state::SpeciesReactionIntegrators::SPECIESREACTIONINT_RK45;
    return S_OK;
}


///////////////////////////////
// state::SpeciesReactionDef //
///////////////////////////////


HRESULT tfStateSpeciesReactionDef_init(struct tfStateSpeciesReactionDefHandle *handle, const char* sbmlInput) {
    TFC_PTRCHECK(handle);
    state::SpeciesReactionDef *tfObj = new state::SpeciesReactionDef(sbmlInput, {});
    handle->tfObj = (void*)tfObj;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_initAntimonyS(struct tfStateSpeciesReactionDefHandle *handle, const char* modelString) {
    TFC_PTRCHECK(handle);
    state::SpeciesReactionDef *tfObj = new state::SpeciesReactionDef(state::SpeciesReactionDef::fromAntimonyString(modelString, {}));
    handle->tfObj = (void*)tfObj;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_initAntimonyF(struct tfStateSpeciesReactionDefHandle *handle, const char* modelFilePath) {
    TFC_PTRCHECK(handle);
    state::SpeciesReactionDef *tfObj = new state::SpeciesReactionDef(state::SpeciesReactionDef::fromAntimonyFile(modelFilePath, {}));
    handle->tfObj = (void*)tfObj;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_copy(struct tfStateSpeciesReactionDefHandle *handle, struct tfStateSpeciesReactionDefHandle *other) {
    TFC_PTRCHECK(handle);
    TFC_SPECIESREACTIONDEF_GET(other, otherTfObj);
    state::SpeciesReactionDef *tfObj = new state::SpeciesReactionDef(*otherTfObj);
    handle->tfObj = (void*)tfObj;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_destroy(struct tfStateSpeciesReactionDefHandle *handle) {
    return TissueForge::capi::destroyHandle<state::SpeciesReactionDef, tfStateSpeciesReactionDefHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfStateSpeciesReactionDef_mapNames(
    struct tfStateSpeciesReactionDefHandle *handle, 
    const char* speciesName, 
    const char* modelName
) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    tfObj->speciesNames.emplace_back(speciesName);
    tfObj->modelNames.emplace_back(modelName);
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_numMappedNames(struct tfStateSpeciesReactionDefHandle *handle, int* result) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    TFC_PTRCHECK(result);
    *result = tfObj->speciesNames.size();
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_mappedNames(
    struct tfStateSpeciesReactionDefHandle *handle, 
    int idx, 
    char** speciesName, 
    unsigned int* numCharsSpeciesName, 
    char** modelName, 
    unsigned int* numCharsModelName
) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    TFC_PTRCHECK(speciesName);
    TFC_PTRCHECK(numCharsSpeciesName);
    TFC_PTRCHECK(modelName);
    TFC_PTRCHECK(numCharsModelName);
    if(idx >= tfObj->speciesNames.size()) 
        return E_FAIL;
    if(TissueForge::capi::str2Char(tfObj->speciesNames[idx], speciesName, numCharsSpeciesName) != S_OK) 
        return E_FAIL;
    if(TissueForge::capi::str2Char(tfObj->modelNames[idx], modelName, numCharsModelName) != S_OK) 
        return E_FAIL;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_getSBMLInput(
    struct tfStateSpeciesReactionDefHandle *handle, 
    char** sbmlInput, 
    unsigned int* numChars
) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    TFC_PTRCHECK(sbmlInput);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(tfObj->uriOrSBML, sbmlInput, numChars);
}

HRESULT tfStateSpeciesReactionDef_setSBMLInput(struct tfStateSpeciesReactionDefHandle *handle, const char* sbmlInput) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    TFC_PTRCHECK(sbmlInput);
    tfObj->uriOrSBML = std::string(sbmlInput);
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_getStepSize(struct tfStateSpeciesReactionDefHandle *handle, double* value) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    TFC_PTRCHECK(value);
    *value = tfObj->stepSize;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_setStepSize(struct tfStateSpeciesReactionDefHandle *handle, double value) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    tfObj->stepSize = value;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_getNumSteps(struct tfStateSpeciesReactionDefHandle *handle, int* value) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    TFC_PTRCHECK(value);
    *value = tfObj->numSteps;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_setNumSteps(struct tfStateSpeciesReactionDefHandle *handle, int value) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    tfObj->numSteps = value;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_getIntegrator(struct tfStateSpeciesReactionDefHandle *handle, int* value) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    TFC_PTRCHECK(value);
    *value = tfObj->integrator;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_setIntegrator(struct tfStateSpeciesReactionDefHandle *handle, int value) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    tfObj->integrator = (state::SpeciesReactionIntegrators)value;
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_integratorName(int sri, char** name, unsigned int* numChars) {
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(state::SpeciesReactionDef::integratorName((state::SpeciesReactionIntegrators)sri), name, numChars);
}

HRESULT tfStateSpeciesReactionDef_integratorEnum(const char* name, int* value) {
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(value);
    *value = state::SpeciesReactionDef::integratorEnum(name);
    return S_OK;
}

HRESULT tfStateSpeciesReactionDef_toString(struct tfStateSpeciesReactionDefHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIESREACTIONDEF_GET(handle, tfObj);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(tfObj->toString(), str, numChars);
}

HRESULT tfStateSpeciesReactionDef_fromString(struct tfStateSpeciesReactionDefHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    state::SpeciesReactionDef *tfObj = new state::SpeciesReactionDef(state::SpeciesReactionDef::fromString(str));
    handle->tfObj = (void*)tfObj;
    return S_OK;
}


////////////////////////////
// state::SpeciesReaction //
////////////////////////////


HRESULT tfStateSpeciesReaction_initD(
    struct tfStateSpeciesReactionHandle *handle, 
    struct tfStateStateVectorHandle *svec, 
    struct tfStateSpeciesReactionDefHandle *rdef, 
    bool mapFrom
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(svec);
    TFC_SPECIESREACTIONDEF_GET(rdef, _rdef);
    state::StateVector *_svec = (state::StateVector*)svec->tfObj;
    TFC_PTRCHECK(_svec);
    state::SpeciesReaction *tfObj = new state::SpeciesReaction(_svec, *_rdef, mapFrom);
    handle->tfObj = (void*)tfObj;
    return S_OK;
}

HRESULT tfStateSpeciesReaction_initS(
    struct tfStateSpeciesReactionHandle *handle, 
    struct tfStateStateVectorHandle *svec,
    const char* sbmlInput
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(svec);
    TFC_PTRCHECK(sbmlInput);
    state::StateVector *_svec = (state::StateVector*)svec->tfObj;
    TFC_PTRCHECK(_svec);
    state::SpeciesReaction *tfObj = new state::SpeciesReaction(_svec, sbmlInput, false);
    handle->tfObj = (void*)tfObj;
    return S_OK;
}

HRESULT tfStateSpeciesReaction_copy(
    struct tfStateSpeciesReactionHandle *handle, 
    struct tfStateStateVectorHandle *svec, 
    struct tfStateSpeciesReactionHandle *other, 
    bool mapFrom
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(svec);
    TFC_SPECIESREACTION_GET(other, _other);
    state::StateVector *_svec = (state::StateVector*)svec->tfObj;
    TFC_PTRCHECK(_svec);
    state::SpeciesReaction *tfObj = new state::SpeciesReaction(_svec, *_other, mapFrom);
    handle->tfObj = (void*)tfObj;
    return S_OK;
}

HRESULT tfStateSpeciesReaction_destroy(struct tfStateSpeciesReactionHandle *handle) {
    return TissueForge::capi::destroyHandle<state::SpeciesReaction, tfStateSpeciesReactionHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfStateSpeciesReaction_getOwner(struct tfStateSpeciesReactionHandle *handle, tfStateStateVectorHandle *svec) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(svec);
    svec->tfObj = (void*)tfObj->owner;
    return S_OK;
}

HRESULT tfStateSpeciesReaction_stepP(struct tfStateSpeciesReactionHandle *handle, double until) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->step(until);
}

HRESULT tfStateSpeciesReaction_step(struct tfStateSpeciesReactionHandle *handle, unsigned int numSteps) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->step(numSteps);
}

HRESULT tfStateSpeciesReaction_stepT(struct tfStateSpeciesReactionHandle *handle, double univdt) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->stepT(univdt);
}

HRESULT tfStateSpeciesReaction_reset(struct tfStateSpeciesReactionHandle *handle) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->reset();
}

HRESULT tfStateSpeciesReaction_mapName(struct tfStateSpeciesReactionHandle *handle, const char* speciesName, const char* modelName) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(speciesName);
    TFC_PTRCHECK(modelName);
    return tfObj->mapName(speciesName, modelName);
}

HRESULT tfStateSpeciesReaction_numMappedNames(struct tfStateSpeciesReactionHandle *handle, int* numNames) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(numNames);
    *numNames = tfObj->getSpeciesNames().size();
    return S_OK;
}

HRESULT tfStateSpeciesReaction_mappedNames(
    struct tfStateSpeciesReactionHandle *handle, 
    int idx, 
    char** speciesName, 
    unsigned int* numCharsSpeciesName, 
    char** modelName, 
    unsigned int* numCharsModelName
) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(speciesName);
    TFC_PTRCHECK(numCharsSpeciesName);
    TFC_PTRCHECK(modelName);
    TFC_PTRCHECK(numCharsModelName);
    if(idx >= tfObj->getSpeciesNames().size()) 
        return E_FAIL;
    if(TissueForge::capi::str2Char(tfObj->getSpeciesNames()[idx], speciesName, numCharsSpeciesName) != S_OK) 
        return E_FAIL;
    if(TissueForge::capi::str2Char(tfObj->getModelNames()[idx], modelName, numCharsModelName) != S_OK) 
        return E_FAIL;
    return S_OK;
}

HRESULT tfStateSpeciesReaction_resetMap(struct tfStateSpeciesReactionHandle *handle) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->resetMap();
}

HRESULT tfStateSpeciesReaction_mapValuesTo(struct tfStateSpeciesReactionHandle *handle) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->mapValuesTo();
}

HRESULT tfStateSpeciesReaction_mapValuesFrom(struct tfStateSpeciesReactionHandle *handle) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->mapValuesFrom();
}

HRESULT tfStateSpeciesReaction_setStepSize(struct tfStateSpeciesReactionHandle *handle, double value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->setStepSize(value);
}

HRESULT tfStateSpeciesReaction_getStepSize(struct tfStateSpeciesReactionHandle *handle, double* value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(value);
    *value = tfObj->getStepSize();
    return S_OK;
}

HRESULT tfStateSpeciesReaction_setNumSteps(struct tfStateSpeciesReactionHandle *handle, int value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->setNumSteps(value);
}

HRESULT tfStateSpeciesReaction_getNumSteps(struct tfStateSpeciesReactionHandle *handle, int* value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(value);
    *value = tfObj->getNumSteps();
    return S_OK;
}

HRESULT tfStateSpeciesReaction_setCurrentTime(struct tfStateSpeciesReactionHandle *handle, double value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->setCurrentTime(value);
}

HRESULT tfStateSpeciesReaction_getCurrentTime(struct tfStateSpeciesReactionHandle *handle, double* value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(value);
    *value = tfObj->getCurrentTime();
    return S_OK;
}

HRESULT tfStateSpeciesReaction_setModelValue(struct tfStateSpeciesReactionHandle *handle, const char* name, double value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(name);
    return tfObj->setModelValue(name, value);
}

HRESULT tfStateSpeciesReaction_getModelValue(struct tfStateSpeciesReactionHandle *handle, const char* name, double* value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(value);
    *value = tfObj->getModelValue(name);
    return S_OK;
}

HRESULT tfStateSpeciesReaction_hasModelValue(struct tfStateSpeciesReactionHandle *handle, const char* name, bool* value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(value);
    *value = tfObj->hasModelValue(name);
    return S_OK;
}

HRESULT tfStateSpeciesReaction_getIntegratorName(struct tfStateSpeciesReactionHandle *handle, char** name, unsigned int* numChars) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(tfObj->getIntegratorName(), name, numChars);
}

HRESULT tfStateSpeciesReaction_setIntegratorName(struct tfStateSpeciesReactionHandle *handle, const char* name) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(name);
    return tfObj->setIntegratorName(name);
}

HRESULT tfStateSpeciesReaction_hasIntegrator(struct tfStateSpeciesReactionHandle *handle, int sri, bool* value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(value);
    *value = tfObj->hasIntegrator((state::SpeciesReactionIntegrators)sri);
    return S_OK;
}

HRESULT tfStateSpeciesReaction_getIntegrator(struct tfStateSpeciesReactionHandle *handle, int* value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(value);
    *value = (int)tfObj->getIntegrator();
    return S_OK;
}

HRESULT tfStateSpeciesReaction_setIntegrator(struct tfStateSpeciesReactionHandle *handle, int value) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    return tfObj->setIntegrator((state::SpeciesReactionIntegrators)value);
}

HRESULT tfStateSpeciesReaction_toString(struct tfStateSpeciesReactionHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIESREACTION_GET(handle, tfObj);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(tfObj->toString(), str, numChars);
}

HRESULT tfStateSpeciesReaction_fromString(struct tfStateSpeciesReactionHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    state::SpeciesReaction *tfObj = new state::SpeciesReaction(state::SpeciesReaction::fromString(str));
    handle->tfObj = (void*)tfObj;
    return S_OK;
}


/////////////////////////////
// state::SpeciesReactions //
/////////////////////////////


HRESULT tfStateSpeciesReactions_init(struct tfStateSpeciesReactionsHandle *handle, tfStateStateVectorHandle *svec) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(svec);
    state::StateVector *_svec = (state::StateVector*)svec->tfObj;
    TFC_PTRCHECK(_svec);
    state::SpeciesReactions *tfObj = new state::SpeciesReactions(_svec);
    handle->tfObj = (void*)tfObj;
    return S_OK;
}

HRESULT tfStateSpeciesReactions_copy(
    struct tfStateSpeciesReactionsHandle *handle, 
    tfStateStateVectorHandle *svec, 
    struct tfStateSpeciesReactionsHandle *other
) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(svec);
    TFC_SPECIESREACTIONS_GET(other, _other);
    state::StateVector *_svec = (state::StateVector*)svec->tfObj;
    TFC_PTRCHECK(_svec);
    state::SpeciesReactions *tfObj = new state::SpeciesReactions(_svec);
    if(tfObj->copyFrom(*_other) != S_OK) {
        delete tfObj;
        return E_FAIL;
    }
    handle->tfObj = (void*)tfObj;
    return S_OK;
}

HRESULT tfStateSpeciesReactions_destroy(struct tfStateSpeciesReactionsHandle *handle) {
    return TissueForge::capi::destroyHandle<state::SpeciesReactions, tfStateSpeciesReactionsHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfStateSpeciesReactions_numModels(struct tfStateSpeciesReactionsHandle *handle, unsigned int* value) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    TFC_PTRCHECK(value);
    *value = tfObj->models.size();
    return S_OK;
}

HRESULT tfStateSpeciesReactions_getModel(
    struct tfStateSpeciesReactionsHandle *handle, 
    unsigned int idx, 
    tfStateSpeciesReactionHandle* model
) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    TFC_PTRCHECK(model);
    if(idx >= tfObj->models.size()) 
        return E_FAIL;
    auto i = tfObj->models.begin();
    i = std::next(i, idx);
    model->tfObj = (void*)i->second;
    return S_OK;
}

HRESULT tfStateSpeciesReactions_getModelName(
    struct tfStateSpeciesReactionsHandle *handle, 
    unsigned int idx, 
    char** name, 
    unsigned int* numChars
) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(numChars);
    if(idx >= tfObj->models.size()) 
        return E_FAIL;
    auto i = tfObj->models.begin();
    i = std::next(i, idx);
    return TissueForge::capi::str2Char(i->first, name, numChars);
}

HRESULT tfStateSpeciesReactions_getModelByName(
    struct tfStateSpeciesReactionsHandle *handle, 
    const char* name, 
    tfStateSpeciesReactionHandle* model
) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    TFC_PTRCHECK(name);
    TFC_PTRCHECK(model);
    auto mitr = tfObj->models.find(name);
    if(mitr == tfObj->models.end()) 
        return E_FAIL;
    model->tfObj = (void*)mitr->second;
    return S_OK;
}

HRESULT tfStateSpeciesReactions_getOwner(struct tfStateSpeciesReactionsHandle *handle, tfStateStateVectorHandle *owner) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    TFC_PTRCHECK(owner);
    owner->tfObj = (void*)tfObj->owner;
    return S_OK;
}

HRESULT tfStateSpeciesReactions_step(struct tfStateSpeciesReactionsHandle *handle, unsigned int numSteps) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    return tfObj->step(numSteps);
}

HRESULT tfStateSpeciesReactions_stepT(struct tfStateSpeciesReactionsHandle *handle, double univdt) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    return tfObj->stepT(univdt);
}

HRESULT tfStateSpeciesReactions_reset(struct tfStateSpeciesReactionsHandle *handle) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    return tfObj->reset();
}

HRESULT tfStateSpeciesReactions_create(
    struct tfStateSpeciesReactionsHandle *handle, 
    const char* modelName, 
    tfStateSpeciesReactionDefHandle* rDef, 
    bool mapFrom
) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    TFC_PTRCHECK(modelName);
    TFC_SPECIESREACTIONDEF_GET(rDef, _rDef);
    return tfObj->create(modelName, *_rDef, mapFrom);
}

HRESULT tfStateSpeciesReactions_createC(
    struct tfStateSpeciesReactionsHandle *handle, 
    const char* modelName, 
    tfStateSpeciesReactionHandle* model, 
    bool mapFrom
) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    TFC_PTRCHECK(modelName);
    TFC_SPECIESREACTION_GET(model, _model);
    return tfObj->create(modelName, *_model, mapFrom);
}

HRESULT tfStateSpeciesReactions_mapValuesTo(struct tfStateSpeciesReactionsHandle *handle) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    return tfObj->mapValuesTo();
}

HRESULT tfStateSpeciesReactions_mapValuesFrom(struct tfStateSpeciesReactionsHandle *handle) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    return tfObj->mapValuesFrom();
}

HRESULT tfStateSpeciesReactions_toString(struct tfStateSpeciesReactionsHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIESREACTIONS_GET(handle, tfObj);
    TFC_PTRCHECK(str);
    TFC_PTRCHECK(numChars);
    return TissueForge::capi::str2Char(tfObj->toString(), str, numChars);
}

HRESULT tfStateSpeciesReactions_fromString(struct tfStateSpeciesReactionsHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    state::SpeciesReactions *tfObj = new state::SpeciesReactions(state::SpeciesReactions::fromString(str));
    handle->tfObj = (void*)tfObj;
    return S_OK;
}
