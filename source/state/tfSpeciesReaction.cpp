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

#include "tfSpeciesReaction.h"
#include "tfStateVector.h"

#include <tfError.h>
#include <tfLogger.h>
#include <tfEngine.h>
#include <io/tfFIO.h>

#include <rrRoadRunner.h>
#include <rrSelectionRecord.h>

#include <algorithm>
#include <unordered_map>


using namespace TissueForge;


state::SpeciesReaction::SpeciesReaction(
    state::StateVector* svec, 
    const std::string& uriOrSBML, 
    const std::vector<std::pair<std::string, std::string > >& nameMap, 
    const bool& mapFrom
) : 
    rr{new rr::RoadRunner(uriOrSBML)},
    owner{svec}
{
    for(auto& nm : nameMap) mapName(std::get<0>(nm), std::get<1>(nm));

    if(mapFrom) mapValuesFrom();
    else mapValuesTo();
}

state::SpeciesReaction::SpeciesReaction(state::StateVector* svec, const std::string& uriOrSBML, const bool& mapFrom) : 
    SpeciesReaction(svec, uriOrSBML, {}, mapFrom)
{}

state::SpeciesReaction::SpeciesReaction(state::StateVector* svec, const state::SpeciesReactionDef& rDef, const bool& mapFrom) : 
    SpeciesReaction()
{
    rr = new rr::RoadRunner(rDef.uriOrSBML);
    owner = svec;

    if(rDef.modelNames.size() != rDef.speciesNames.size()) { tf_error(E_FAIL, "Mapping is inconsistent"); } 
    else for(int i = 0; i < rDef.modelNames.size(); i++) mapName(rDef.speciesNames[i], rDef.modelNames[i]);

    dt = rDef.stepSize;
    numSteps = rDef.numSteps;
    setIntegrator(rDef.integrator);

    if(mapFrom) mapValuesFrom();
    else mapValuesTo();
}

state::SpeciesReaction::SpeciesReaction(state::StateVector* svec, const state::SpeciesReaction& other, const bool& mapFrom) : 
    SpeciesReaction(svec, other.rr->getSBML())
{
    try { rr->loadStateS(other.rr->saveStateS()); } catch(const rr::Exception& e) { tf_error(E_FAIL, e.getMessage().c_str()); }
    if(mapFrom) mapValuesFrom();
    else mapValuesTo();

    svecIndices = other.svecIndices;
    modelNames = other.modelNames;
    currentTime = other.currentTime;
    dt = other.dt;
    numSteps = other.numSteps;
}

state::SpeciesReaction::~SpeciesReaction() {
    if(rr) {
        delete rr;
        rr = 0;
    }
    owner = 0;
}

HRESULT state::SpeciesReaction::step(const double& until) {
    double _finalTime = until > 0 ? until - currentTime : currentTime + dt;
    double _currentTime = 0.0;

    if(mapValuesTo() != S_OK || resetCachedFluxes() != S_OK) 
        return E_FAIL;

    std::vector<double> oldVals = rr->getSelectedValues();
    TF_Log(LOG_INFORMATION) << oldVals[0];
    
    while(_currentTime < _finalTime) {
        double nextTime = std::min(_currentTime + dt, _finalTime);
        try { rr->simulate(_currentTime, nextTime, numSteps); } catch(const rr::Exception& e) { return tf_error(E_FAIL, e.getMessage().c_str()); }
        _currentTime = nextTime;
    }

    currentTime = until;

    std::vector<double> newVals = rr->getSelectedValues();
    for(int i = 0; i < svecIndices.size(); i++) 
        cachedFluxes[svecIndices[i]] = newVals[i] - oldVals[i];

    return S_OK;
}

HRESULT state::SpeciesReaction::step(const unsigned int& _numSteps) {
    return step(currentTime + dt * _numSteps);
}

HRESULT state::SpeciesReaction::stepT(const double& univdt) {
    static const double iunivdt = 1 / _Engine.dt;

    return step(currentTime + univdt * iunivdt * dt);
}

std::vector<FloatP_t> state::SpeciesReaction::getCachedFluxes() {
    return cachedFluxes;
}

HRESULT state::SpeciesReaction::resetCachedFluxes() {
    cachedFluxes = std::vector<FloatP_t>(owner->size, FPTYPE_ZERO);
    return S_OK;
}

HRESULT state::SpeciesReaction::reset() {
    try { this->rr->loadStateS(this->rr->saveStateS()); } catch(const rr::Exception& e) { return tf_error(E_FAIL, e.getMessage().c_str()); }
    return this->mapValuesFrom();
}

HRESULT state::SpeciesReaction::mapName(const std::string& speciesName, const std::string& modelName) {
    if(std::find(modelNames.begin(), modelNames.end(), modelName) != modelNames.end()) {
        return tf_error(E_FAIL, "Model name already mapped");
    }

    int32_t speciesIndex = owner->species->index_of(speciesName);
    if(speciesIndex < 0) {
        return tf_error(E_FAIL, "Species name not recognized");
    }

    if(std::find(svecIndices.begin(), svecIndices.end(), speciesIndex) != svecIndices.end()) {
        return tf_error(E_FAIL, "Species already mapped");
    }

    svecIndices.push_back(speciesIndex);
    modelNames.push_back(modelName);

    try { rr->setSelections(modelNames); } catch(const rr::Exception& e) { return tf_error(E_FAIL, e.getMessage().c_str()); }
    return S_OK;
}

HRESULT state::SpeciesReaction::mapName(const std::string& speciesName) {
    return mapName(speciesName, speciesName);
}

std::vector<std::string> state::SpeciesReaction::getSpeciesNames() {
    std::vector<std::string> result;
    for(auto& idx : svecIndices) 
        result.push_back(owner->species->item(idx)->getId());
    return result;
}

std::vector<std::string> state::SpeciesReaction::getModelNames() {
    return modelNames;
}

HRESULT state::SpeciesReaction::resetMap() {
    svecIndices.clear();
    modelNames.clear();
    return S_OK;
}

HRESULT state::SpeciesReaction::mapValuesTo() {
    for(int i = 0; i < svecIndices.size(); i++) {
        try{
            rr->setValue(modelNames[i], owner->fvec[svecIndices[i]]);
        }
        catch(const rr::Exception& e) {
            return tf_error(E_FAIL, e.getMessage().c_str());
        }
    }
    return S_OK;
}

HRESULT state::SpeciesReaction::mapValuesFrom() {
    std::vector<double> values = rr->getSelectedValues();
    for(int i = 0; i < svecIndices.size(); i++) 
        owner->fvec[svecIndices[i]] = values[i];
    return S_OK;
}

HRESULT state::SpeciesReaction::setStepSize(const double& stepSize) {
    dt = stepSize;
    return S_OK;
}

double state::SpeciesReaction::getStepSize() {
    return dt;
}

HRESULT state::SpeciesReaction::setNumSteps(const int& _numSteps) {
    numSteps = _numSteps;
    return S_OK;
}

int state::SpeciesReaction::getNumSteps() {
    return numSteps;
}

HRESULT state::SpeciesReaction::setCurrentTime(const double& _currentTime) {
    currentTime = _currentTime;
    return S_OK;
}

double state::SpeciesReaction::getCurrentTime() {
    return currentTime;
}

HRESULT state::SpeciesReaction::setModelValue(const std::string& name, const double& value) {
    try {
        this->rr->setValue(name, value);
        return S_OK;
    }
    catch(const rr::Exception& e) {
        return tf_error(E_FAIL, e.getMessage().c_str());
    }
}

double state::SpeciesReaction::getModelValue(const std::string& name) {
    try {
        return this->rr->getValue(name);
    }
    catch(const rr::Exception& e) {
        tf_error(E_FAIL, e.getMessage().c_str());
        return 0.0;
    }
}

bool state::SpeciesReaction::hasModelValue(const std::string& name) {
    try {
        this->rr->getValue(name);
        return true;
    }
    catch(const rr::Exception& e) {
        return false;
    }
}

std::string state::SpeciesReaction::getIntegratorName() {
    auto i = rr->getIntegrator();
    return i ? i->getName() : SpeciesReactionDef::integratorName(SpeciesReactionIntegrators::SPECIESREACTIONINT_NONE);
}

HRESULT state::SpeciesReaction::setIntegratorName(const std::string& name) {
    SpeciesReactionIntegrators sri = SpeciesReactionDef::integratorEnum(name);
    if(sri == SpeciesReactionIntegrators::SPECIESREACTIONINT_NONE) 
        return tf_error(E_FAIL, "Invalid integrator selection");
    return setIntegrator(sri);
}

bool state::SpeciesReaction::hasIntegrator(const SpeciesReactionIntegrators& sri) {
    return getIntegratorName() == SpeciesReactionDef::integratorName(sri);
}

state::SpeciesReactionIntegrators state::SpeciesReaction::getIntegrator() {
    return SpeciesReactionDef::integratorEnum(getIntegratorName());
}

HRESULT state::SpeciesReaction::setIntegrator(const SpeciesReactionIntegrators& sri) {
    if(sri == SpeciesReactionIntegrators::SPECIESREACTIONINT_NONE) {
        return tf_error(E_FAIL, "Invalid integrator selection");
    }

    rr->setIntegrator(SpeciesReactionDef::integratorName(sri));
    if(!rr->getIntegrator()) {
        return tf_error(E_FAIL, "Integrator selection failed.");
    }
    return S_OK;
}

std::string state::SpeciesReaction::toString() {
    return io::toString(*this);
}

state::SpeciesReaction state::SpeciesReaction::fromString(const std::string &str) {
    return io::fromString<state::SpeciesReaction>(str);
}


namespace TissueForge::io { 

    template<> 
    HRESULT toFile(rr::RoadRunner *dataElement, const MetaData &metaData, IOElement &fileElement) {
        std::stringstream* ss = dataElement->saveStateS();
        TF_IOTOEASY(fileElement, metaData, "state", ss->str());
        delete ss;

        fileElement.get()->type = "RoadRunner";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, rr::RoadRunner *dataElement) {
        std::string state;
        TF_IOFROMEASY(fileElement, metaData, "state", &state);
        std::stringstream *ss = new std::stringstream();
        *ss << state;

        *dataElement = rr::RoadRunner();
        dataElement->loadStateS(ss);

        return S_OK;
    }

    template <>
    HRESULT toFile(const state::SpeciesReaction &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOTOEASY(fileElement, metaData, "rr", dataElement.rr);
        TF_IOTOEASY(fileElement, metaData, "svecIndices", dataElement.svecIndices);
        TF_IOTOEASY(fileElement, metaData, "modelNames", dataElement.modelNames);
        TF_IOTOEASY(fileElement, metaData, "currentTime", dataElement.currentTime);
        TF_IOTOEASY(fileElement, metaData, "dt", dataElement.dt);
        TF_IOTOEASY(fileElement, metaData, "numSteps", dataElement.numSteps);

        fileElement.get()->type = "SpeciesReaction";

        return S_OK;
    }

    template <>
    HRESULT toFile(state::SpeciesReaction *dataElement, const MetaData &metaData, IOElement &fileElement) {
        return toFile(*dataElement, metaData, fileElement);
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, state::SpeciesReaction *dataElement) {
        *dataElement = state::SpeciesReaction();
        dataElement->owner = NULL;
        dataElement->rr = new rr::RoadRunner();

        TF_IOFROMEASY(fileElement, metaData, "rr", dataElement->rr);
        TF_IOFROMEASY(fileElement, metaData, "svecIndices", &dataElement->svecIndices);
        TF_IOFROMEASY(fileElement, metaData, "modelNames", &dataElement->modelNames);
        TF_IOFROMEASY(fileElement, metaData, "currentTime", &dataElement->currentTime);
        TF_IOFROMEASY(fileElement, metaData, "dt", &dataElement->dt);
        TF_IOFROMEASY(fileElement, metaData, "numSteps", &dataElement->numSteps);

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, state::SpeciesReaction **dataElement) {
        *dataElement = new state::SpeciesReaction();
        return fromFile(fileElement, metaData, *dataElement);
    }

};
