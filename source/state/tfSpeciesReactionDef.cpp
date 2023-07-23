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

#include "tfSpeciesReactionDef.h"

#include <tfError.h>
#include <tfLogger.h>
#include <io/tfFIO.h>

#include <antimony_api.h>

#include <unordered_map>


using namespace TissueForge;


static std::string SpeciesReactionInt_NameNone = "none";
static std::unordered_map<state::SpeciesReactionIntegrators, std::string> integratorMap = {
    {state::SpeciesReactionIntegrators::SPECIESREACTIONINT_NONE, SpeciesReactionInt_NameNone},
    {state::SpeciesReactionIntegrators::SPECIESREACTIONINT_CVODE, "cvode"},
    {state::SpeciesReactionIntegrators::SPECIESREACTIONINT_EULER, "euler"},
    {state::SpeciesReactionIntegrators::SPECIESREACTIONINT_GILLESPIE, "gillespie"},
    {state::SpeciesReactionIntegrators::SPECIESREACTIONINT_NLEQ1, "nleq1"},
    {state::SpeciesReactionIntegrators::SPECIESREACTIONINT_NLEQ2, "nleq2"},
    {state::SpeciesReactionIntegrators::SPECIESREACTIONINT_RK4, "rk4"},
    {state::SpeciesReactionIntegrators::SPECIESREACTIONINT_RK45, "rk45"}
};


static std::string AntimonyGetLoadedSBML() {
    char* moduleName = getMainModuleName();
    if(!moduleName) {
        TF_Log(LOG_ERROR) << getLastError();
        tf_error(E_FAIL, "Error fetching Antimony model");
        return {};
    }

    return std::string(getSBMLString(moduleName));
}


static std::string AntimonyFileToSBML(const char* antimonyFilePath) {
    clearPreviousLoads();

    // load file

    if(loadAntimonyFile(antimonyFilePath) < 0) {
        TF_Log(LOG_ERROR) << getLastError();
        tf_error(E_FAIL, "Error loading Antimony model");
        return {};
    }

    return AntimonyGetLoadedSBML();
}

static std::string AntimonyStringToSBML(const char* antimonyString) {
    clearPreviousLoads();

    // load string

    if(loadAntimonyString(antimonyString) < 0) {
        TF_Log(LOG_ERROR) << getLastError();
        tf_error(E_FAIL, "Error loading Antimony model");
        return {};
    }

    return AntimonyGetLoadedSBML();
}


state::SpeciesReactionDef::SpeciesReactionDef(
    std::string _uriOrSBML, 
    const std::vector<std::pair<std::string, std::string > >& nameMap
) : 
    uriOrSBML{_uriOrSBML}
{
    for(auto &nn : nameMap) {
        std::string speciesName, modelName;
        std::tie(speciesName, modelName) = nn;
        speciesNames.push_back(speciesName);
        modelNames.push_back(modelName);
    }
}

state::SpeciesReactionDef state::SpeciesReactionDef::fromAntimonyString(
    const std::string& modelString, 
    const std::vector<std::pair<std::string, std::string > >& nameMap
) {
    return SpeciesReactionDef(AntimonyStringToSBML(modelString.c_str()), nameMap);
}

state::SpeciesReactionDef state::SpeciesReactionDef::fromAntimonyFile(
    const std::string& modelFilePath, 
    const std::vector<std::pair<std::string, std::string > >& nameMap
) {
    return SpeciesReactionDef(AntimonyFileToSBML(modelFilePath.c_str()), nameMap);
}

std::string state::SpeciesReactionDef::integratorName(const SpeciesReactionIntegrators& sri) {
    for(auto& mitr : integratorMap) 
        if(mitr.first == sri) 
            return mitr.second;
    return SpeciesReactionInt_NameNone;
}

state::SpeciesReactionIntegrators state::SpeciesReactionDef::integratorEnum(const std::string& name) {
    for(auto& mitr : integratorMap) 
        if(mitr.second == name) 
            return mitr.first;
    return SpeciesReactionIntegrators::SPECIESREACTIONINT_NONE;
}

std::string state::SpeciesReactionDef::toString() {
    return io::toString(*this);
}

state::SpeciesReactionDef state::SpeciesReactionDef::fromString(const std::string &str) {
    return io::fromString<state::SpeciesReactionDef>(str);
}


namespace TissueForge::io { 

    template <>
    HRESULT toFile(const state::SpeciesReactionDef &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOTOEASY(fileElement, metaData, "uriOrSBML", dataElement.uriOrSBML);
        TF_IOTOEASY(fileElement, metaData, "speciesNames", dataElement.speciesNames);
        TF_IOTOEASY(fileElement, metaData, "modelNames", dataElement.modelNames);
        TF_IOTOEASY(fileElement, metaData, "stepSize", dataElement.stepSize);
        TF_IOTOEASY(fileElement, metaData, "numSteps", dataElement.numSteps);
        TF_IOTOEASY(fileElement, metaData, "integrator", (unsigned int)dataElement.integrator);

        fileElement.get()->type = "SpeciesReactionDef";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, state::SpeciesReactionDef *dataElement) {
        *dataElement = state::SpeciesReactionDef();

        TF_IOFROMEASY(fileElement, metaData, "uriOrSBML", &dataElement->uriOrSBML);
        TF_IOFROMEASY(fileElement, metaData, "speciesNames", &dataElement->speciesNames);
        TF_IOFROMEASY(fileElement, metaData, "modelNames", &dataElement->modelNames);
        TF_IOFROMEASY(fileElement, metaData, "stepSize", &dataElement->stepSize);
        TF_IOFROMEASY(fileElement, metaData, "numSteps", &dataElement->numSteps);

        unsigned int integrator;
        TF_IOFROMEASY(fileElement, metaData, "integrator", &integrator);
        dataElement->integrator = (state::SpeciesReactionIntegrators)integrator;

        return S_OK;
    }

};
