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

/**
 * @file tfCSpeciesReaction.h
 * 
 */

#ifndef _WRAPS_C_TFCSPECIESREACTION_H_
#define _WRAPS_C_TFCSPECIESREACTION_H_

#include "tf_port_c.h"

#include "tfCStateVector.h"

// Handles

/**
 * @brief Reaction integrator enums
 * 
*/
struct CAPI_EXPORT tfStateSpeciesReactionIntegratorsEnum {
    int SPECIESREACTIONINT_NONE;
    int SPECIESREACTIONINT_CVODE;
    int SPECIESREACTIONINT_EULER;
    int SPECIESREACTIONINT_GILLESPIE;
    int SPECIESREACTIONINT_NLEQ1;
    int SPECIESREACTIONINT_NLEQ2;
    int SPECIESREACTIONINT_RK4;
    int SPECIESREACTIONINT_RK45;
};

/**
 * @brief Handle to a @ref state::SpeciesReactionDef instance
 * 
 */
struct CAPI_EXPORT tfStateSpeciesReactionDefHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref state::SpeciesReaction instance
 * 
 */
struct CAPI_EXPORT tfStateSpeciesReactionHandle {
    void *tfObj;
};

/**
 * @brief Handle to a @ref state::SpeciesReactions instance
 * 
 */
struct CAPI_EXPORT tfStateSpeciesReactionsHandle {
    void *tfObj;
};


///////////////////////////////////////
// state::SpeciesReactionIntegrators //
///////////////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionIntegratorsEnum_init(struct tfStateSpeciesReactionIntegratorsEnum* handle);


///////////////////////////////
// state::SpeciesReactionDef //
///////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param sbmlInput SBML model string or file path
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_init(struct tfStateSpeciesReactionDefHandle *handle, const char* sbmlInput);

/**
 * @brief Initialize an instance using an Antimony model string
 * 
 * @param handle handle to populate
 * @param modelString Antimony model string
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_initAntimonyS(struct tfStateSpeciesReactionDefHandle *handle, const char* modelString);

/**
 * @brief Initialize an instance using an Antimony model file
 * 
 * @param handle handle to populate
 * @param modelFilePath Antimony model file
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_initAntimonyF(struct tfStateSpeciesReactionDefHandle *handle, const char* modelFilePath);

/**
 * @brief Initialize an instance from another instance
 * 
 * @param handle handle to populate
 * @param other populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_copy(struct tfStateSpeciesReactionDefHandle *handle, struct tfStateSpeciesReactionDefHandle *other);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_destroy(struct tfStateSpeciesReactionDefHandle *handle);

/**
 * @brief Map a species name to a model variable name
 * 
 * @param handle populated handle
 * @param speciesName species name
 * @param modelName model variable name
 * @return S_OK on success
*/
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_mapNames(
    struct tfStateSpeciesReactionDefHandle *handle, 
    const char* speciesName, 
    const char* modelName
);

/**
 * @brief Get the number of mapped names
 * 
 * @param handle populated handle
 * @param result number of mapped names
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_numMappedNames(struct tfStateSpeciesReactionDefHandle *handle, int* result);

/**
 * @brief Get the mapped names by index
 * 
 * @param handle populated handle
 * @param idx index
 * @param speciesName species name
 * @param numCharsSpeciesName number of species name characters
 * @param modelName model variable name
 * @param numCharsModelName number of model variable name characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_mappedNames(
    struct tfStateSpeciesReactionDefHandle *handle, 
    int idx, 
    char** speciesName, 
    unsigned int* numCharsSpeciesName, 
    char** modelName, 
    unsigned int* numCharsModelName
);

/**
 * @brief Get the SBML model string or file path
 * 
 * @param handle populated handle
 * @param sbmlInput SBML model string or file path
 * @param numChars number of characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_getSBMLInput(
    struct tfStateSpeciesReactionDefHandle *handle, 
    char** sbmlInput, 
    unsigned int* numChars
);

/**
 * @brief Set the SBML model string or file path
 * 
 * @param handle populated handle
 * @param sbmlInput SBML model string or file path
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_setSBMLInput(struct tfStateSpeciesReactionDefHandle *handle, const char* sbmlInput);

/**
 * @brief Get the period of a simulation step according to the model
 * 
 * @param handle populated handle
 * @param value period of a simulation step according to the model
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_getStepSize(struct tfStateSpeciesReactionDefHandle *handle, double* value);

/**
 * @brief Set the period of a simulation step according to the model
 * 
 * @param handle populated handle
 * @param value period of a simulation step according to the model
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_setStepSize(struct tfStateSpeciesReactionDefHandle *handle, double value);

/**
 * @brief Get the number of substeps per simulation step
 * 
 * @param handle populated handle
 * @param value number of substeps per simulation step
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_getNumSteps(struct tfStateSpeciesReactionDefHandle *handle, int* value);

/**
 * @brief Set the number of substeps per simulation step
 * 
 * @param handle populated handle
 * @param value number of substeps per simulation step
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_setNumSteps(struct tfStateSpeciesReactionDefHandle *handle, int value);

/**
 * @brief Get the simulation integrator enum
 * 
 * @param handle populated handle
 * @param value simulation integrator enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_getIntegrator(struct tfStateSpeciesReactionDefHandle *handle, int* value);

/**
 * @brief Set the simulation integrator enum
 * 
 * @param handle populated handle
 * @param value simulation integrator enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_setIntegrator(struct tfStateSpeciesReactionDefHandle *handle, int value);

/**
 * @brief Get the name of an integrator by enum
 * 
 * @param sri integrator enum
 * @param name integrator name
 * @param numChars number of integrator name characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_integratorName(int sri, char** name, unsigned int* numChars);

/**
 * @brief Get the enum of an integrator by name
 * 
 * @param name integrator name
 * @param value integrator enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_integratorEnum(const char* name, int* value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_toString(struct tfStateSpeciesReactionDefHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactionDef_fromString(struct tfStateSpeciesReactionDefHandle *handle, const char *str);


////////////////////////////
// state::SpeciesReaction //
////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param svec owner
 * @param rdef model definition
 * @param mapFrom flag to copy values from model to species. false performs the opposite. 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_initD(
    struct tfStateSpeciesReactionHandle *handle, 
    struct tfStateStateVectorHandle *svec,
    struct tfStateSpeciesReactionDefHandle *rdef,
    bool mapFrom
);

/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param svec owner
 * @param sbmlInput SBML model string or file path
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_initS(
    struct tfStateSpeciesReactionHandle *handle, 
    struct tfStateStateVectorHandle *svec,
    const char* sbmlInput
);

/**
 * @brief Initialize an instance from another instance
 * 
 * @param handle handle to populate
 * @param other populated handle
 * @param svec owner
 * @param mapFrom flag to copy values from model to species. false performs the opposite. 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_copy(
    struct tfStateSpeciesReactionHandle *handle, 
    struct tfStateStateVectorHandle *svec, 
    struct tfStateSpeciesReactionHandle *other, 
    bool mapFrom
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_destroy(struct tfStateSpeciesReactionHandle *handle);

/**
 * @brief Get the owner
 * 
 * @param handle populated handle
 * @param svec owner 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_getOwner(struct tfStateSpeciesReactionHandle *handle, tfStateStateVectorHandle *svec);

/**
 * @brief Integrate over a period
 * 
 * @param handle populated handle
 * @param until period over which to integrate the model. Uses the universe time step when <= 0
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_stepP(struct tfStateSpeciesReactionHandle *handle, double until);

/**
 * @brief Integrate over a number of steps
 * 
 * @param handle populated handle
 * @param numSteps Number of steps over which to integrate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_step(struct tfStateSpeciesReactionHandle *handle, unsigned int numSteps);

/**
 * @brief Integrate over a period of time according to the universe
 * 
 * @param handle populated handle
 * @param univdt Period of universe time over which to integrate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_stepT(struct tfStateSpeciesReactionHandle *handle, double univdt);

/**
 * @brief Reset the model to its initial state.
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_reset(struct tfStateSpeciesReactionHandle *handle);

/**
 * @brief Map a species name to a model variable name
 * 
 * @param handle populated handle
 * @param speciesName Species name
 * @param modelName Model variable name
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_mapName(struct tfStateSpeciesReactionHandle *handle, const char* speciesName, const char* modelName);

/**
 * @brief Get number of mapped names
 * 
 * @param handle populated handle
 * @param numNames number of mapped names
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_numMappedNames(struct tfStateSpeciesReactionHandle *handle, int* numNames);

/**
 * @brief Get mapped names
 * 
 * @param handle populated handle
 * @param idx name index
 * @param speciesName species name
 * @param numCharsSpeciesName number of species name characters
 * @param modelName model name
 * @param numCharsModelName number of model name characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_mappedNames(
    struct tfStateSpeciesReactionHandle *handle, 
    int idx, 
    char** speciesName, 
    unsigned int* numCharsSpeciesName, 
    char** modelName, 
    unsigned int* numCharsModelName
);

/**
 * @brief Reset name mapping
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_resetMap(struct tfStateSpeciesReactionHandle *handle);

/**
 * @brief Map values from species to model variables
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_mapValuesTo(struct tfStateSpeciesReactionHandle *handle);

/**
 * @brief Map values from species to model variables
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_mapValuesFrom(struct tfStateSpeciesReactionHandle *handle);

/**
 * @brief Set the period of a step according to the model
 * 
 * @param handle populated handle
 * @param value period of a step according to the model
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_setStepSize(struct tfStateSpeciesReactionHandle *handle, double value);

/**
 * @brief Get the period of a step according to the model
 * 
 * @param handle populated handle
 * @param value period of a step according to the model
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_getStepSize(struct tfStateSpeciesReactionHandle *handle, double* value);

/**
 * @brief Set the number of substeps per simulation step
 * 
 * @param handle populated handle
 * @param value number of substeps per simulation step
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_setNumSteps(struct tfStateSpeciesReactionHandle *handle, int value);

/**
 * @brief Get the number of substeps per simulation step
 * 
 * @param handle populated handle
 * @param value number of substeps per simulation step
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_getNumSteps(struct tfStateSpeciesReactionHandle *handle, int* value);

/**
 * @brief Set the current time according to the model
 * 
 * @param handle populated handle
 * @param value current time according to the model
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_setCurrentTime(struct tfStateSpeciesReactionHandle *handle, double value);

/**
 * @brief Get the current time according to the model
 * 
 * @param handle populated handle
 * @param value current time according to the model
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_getCurrentTime(struct tfStateSpeciesReactionHandle *handle, double* value);

/**
 * @brief Set the value of a model variable by name
 * 
 * @param handle populated handle
 * @param name model variable name
 * @param value Model variable value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_setModelValue(struct tfStateSpeciesReactionHandle *handle, const char* name, double value);

/**
 * @brief Get the value of a model variable by name
 * 
 * @param handle populated handle
 * @param name model variable name
 * @param value model variable value
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_getModelValue(struct tfStateSpeciesReactionHandle *handle, const char* name, double* value);

/**
 * @brief Test whether the model has a variable by name
 * 
 * @param handle populated handle
 * @param name model variable name
 * @param value flag indicating whether the model has the variable
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_hasModelValue(struct tfStateSpeciesReactionHandle *handle, const char* name, bool* value);

/**
 * @brief Get the name of the current integrator
 * 
 * @param handle populated handle
 * @param name name of the current integrator
 * @param numChars number of name characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_getIntegratorName(struct tfStateSpeciesReactionHandle *handle, char** name, unsigned int* numChars);

/**
 * @brief Set the current integrator by name
 * 
 * @param handle populated handle
 * @param name integrator name
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_setIntegratorName(struct tfStateSpeciesReactionHandle *handle, const char* name);

/**
 * @brief Test whether using an integrator by enum
 * 
 * @param handle populated handle
 * @param sri integrator enum to test
 * @param value flag indicating whether using the integrator
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_hasIntegrator(struct tfStateSpeciesReactionHandle *handle, int sri, bool* value);

/**
 * @brief Get the current integrator enum
 * 
 * @param handle populated handle
 * @param value current integrator enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_getIntegrator(struct tfStateSpeciesReactionHandle *handle, int* value);

/**
 * @brief Set the current integrator by enum
 * 
 * @param handle populated handle
 * @param value integrator enum
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_setIntegrator(struct tfStateSpeciesReactionHandle *handle, int value);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_toString(struct tfStateSpeciesReactionHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReaction_fromString(struct tfStateSpeciesReactionHandle *handle, const char *str);


/////////////////////////////
// state::SpeciesReactions //
/////////////////////////////


/**
 * @brief Initialize an instance
 * 
 * @param handle handle to populate
 * @param svec owner
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_init(struct tfStateSpeciesReactionsHandle *handle, tfStateStateVectorHandle *svec);

/**
 * @brief Initialize an instance from another instance
 * 
 * @param handle handle to populate
 * @param svec owner
 * @param other populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_copy(
    struct tfStateSpeciesReactionsHandle *handle, 
    tfStateStateVectorHandle *svec, 
    struct tfStateSpeciesReactionsHandle *other
);

/**
 * @brief Destroy an instance
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_destroy(struct tfStateSpeciesReactionsHandle *handle);

/**
 * @brief Get the number of models
 * 
 * @param handle populated handle
 * @param value number of models
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_numModels(struct tfStateSpeciesReactionsHandle *handle, unsigned int* value);

/**
 * @brief Get a model by index
 * 
 * @param handle populated handle
 * @param idx index
 * @param model model
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_getModel(
    struct tfStateSpeciesReactionsHandle *handle, 
    unsigned int idx, 
    tfStateSpeciesReactionHandle* model
);

/**
 * @brief Get a model name
 * 
 * @param handle populated handle
 * @param idx index
 * @param name model name
 * @param numChars number of model name characters
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_getModelName(
    struct tfStateSpeciesReactionsHandle *handle, 
    unsigned int idx, 
    char** name, 
    unsigned int* numChars
);

/**
 * @brief Get a model by name
 * 
 * @param handle populated handle
 * @param name model name
 * @param model model
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_getModelByName(
    struct tfStateSpeciesReactionsHandle *handle, 
    const char* name, 
    tfStateSpeciesReactionHandle* model
);

/**
 * @brief Get the owner
 * 
 * @param handle populated handle
 * @param owner owner
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_getOwner(struct tfStateSpeciesReactionsHandle *handle, tfStateStateVectorHandle *owner);

/**
 * @brief Integrate all models a number of steps
 * 
 * @param handle populated handle
 * @param numSteps number of steps over which to integrate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_step(struct tfStateSpeciesReactionsHandle *handle, unsigned int numSteps);

/**
 * @brief Integrate all models over a period
 * 
 * @param handle populated handle
 * @param univdt period of time according to the universe over which to integrate
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_stepT(struct tfStateSpeciesReactionsHandle *handle, double univdt);

/**
 * @brief Reset all models to the initial state
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_reset(struct tfStateSpeciesReactionsHandle *handle);

/**
 * @brief Create a model
 * 
 * @param handle populated handle
 * @param modelName model name
 * @param rDef model definition
 * @param mapFrom flag to map values from the model to species when true. false performs the opposite. 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_create(
    struct tfStateSpeciesReactionsHandle *handle, 
    const char* modelName, 
    tfStateSpeciesReactionDefHandle* rDef, 
    bool mapFrom
);

/**
 * @brief Create a model
 * 
 * @param handle populated handle
 * @param modelName model name
 * @param model model to copy
 * @param mapFrom flag to map values from the model to species when true. false performs the opposite. 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_createC(
    struct tfStateSpeciesReactionsHandle *handle, 
    const char* modelName, 
    tfStateSpeciesReactionHandle* model, 
    bool mapFrom
);

/**
 * @brief Map values from species to model variables
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_mapValuesTo(struct tfStateSpeciesReactionsHandle *handle);

/**
 * @brief Map values from species to model variables
 * 
 * @param handle populated handle
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_mapValuesFrom(struct tfStateSpeciesReactionsHandle *handle);

/**
 * @brief Get a JSON string representation
 * 
 * @param handle populated handle
 * @param str string representation
 * @param numChars number of characters of string representation
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_toString(struct tfStateSpeciesReactionsHandle *handle, char **str, unsigned int *numChars);

/**
 * @brief Create from a JSON string representation. 
 * 
 * @return S_OK on success
 */
CAPI_FUNC(HRESULT) tfStateSpeciesReactions_fromString(struct tfStateSpeciesReactionsHandle *handle, const char *str);

#endif // _WRAPS_C_TFCSPECIESREACTION_H_