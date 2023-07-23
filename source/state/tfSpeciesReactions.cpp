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

#include "tfSpeciesReactions.h"
#include "tfStateVector.h"

#include <tfError.h>
#include <tfLogger.h>
#include <io/tfFIO.h>
#include <mdcore_config.h>

#include <rrRoadRunner.h>
#include <rrSelectionRecord.h>


using namespace TissueForge;


state::SpeciesReactions::SpeciesReactions(state::StateVector* svec) : 
    owner{svec}
{}

state::SpeciesReactions::~SpeciesReactions() {
    for(auto& mitr : models) {
        delete mitr.second;
    }
    models.clear();
}

HRESULT state::SpeciesReactions::step(const unsigned int& numSteps) {
    for(auto& mitr : models) 
        if(mitr.second->step(numSteps) != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT state::SpeciesReactions::stepT(const double& univdt) {
    for(auto& mitr : models) 
        if(mitr.second->stepT(univdt) != S_OK) 
            return E_FAIL;
    return S_OK;
}

std::vector<FloatP_t> state::SpeciesReactions::getCachedFluxes() {
    std::vector<FloatP_t> cachedFluxes(owner->size, FPTYPE_ZERO);

    for(auto& mitr : models) {
        std::vector<FloatP_t> m_cachedFluxes = mitr.second->getCachedFluxes();
        for(int i = 0; i < m_cachedFluxes.size(); i++) 
            cachedFluxes[i] += m_cachedFluxes[i];
    }

    return cachedFluxes;
}

HRESULT state::SpeciesReactions::reset() {
    for(auto& mitr : models) 
        if(mitr.second->reset() != S_OK) 
            return E_FAIL;
    return S_OK;
}

HRESULT state::SpeciesReactions::create(const std::string& modelName, const state::SpeciesReactionDef& rDef, const bool& mapFrom) {
    if(models.find(modelName) != models.end()) {
        return tf_error(E_FAIL, "Reaction already registered with this name.");
    }

    SpeciesReaction* reaction = new SpeciesReaction(owner, rDef, mapFrom);
    models.insert({modelName, reaction});
    return S_OK;
}

HRESULT state::SpeciesReactions::create(const std::string& modelName, const SpeciesReaction& model, const bool& mapFrom) {
    if(models.find(modelName) != models.end()) {
        return tf_error(E_FAIL, "Reaction already registered with this name.");
    }

    SpeciesReaction* reaction = new SpeciesReaction(owner, model, mapFrom);
    models.insert({modelName, reaction});
    return S_OK;
}

HRESULT state::SpeciesReactions::copyFrom(const SpeciesReactions& other) {
    for(auto &m : other.models) 
        create(m.first, *m.second);
    return S_OK;
}

HRESULT state::SpeciesReactions::mapValuesTo() {
    for(auto& mitr : models) 
        if(mitr.second->mapValuesTo() != S_OK) {
            return E_FAIL;
        }
    return S_OK;
}

HRESULT state::SpeciesReactions::mapValuesFrom() {
    for(auto& mitr : models) 
        if(mitr.second->mapValuesFrom() != S_OK) {
            return E_FAIL;
        }
    return S_OK;
}

std::string state::SpeciesReactions::toString() {
    return io::toString(*this);
}

state::SpeciesReactions state::SpeciesReactions::fromString(const std::string &str) {
    return io::fromString<state::SpeciesReactions>(str);
}


namespace TissueForge::io { 

    template <>
    HRESULT toFile(const state::SpeciesReactions &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOTOEASY(fileElement, metaData, "models", dataElement.models);

        fileElement.get()->type = "SpeciesReactions";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, state::SpeciesReactions *dataElement) {
        *dataElement = state::SpeciesReactions();
        dataElement->owner = NULL;

        TF_IOFROMEASY(fileElement, metaData, "models", &dataElement->models);

        return S_OK;
    }

};
