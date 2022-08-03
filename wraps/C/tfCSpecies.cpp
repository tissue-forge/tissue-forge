/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2022 T.J. Sego
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

#include "tfCSpecies.h"

#include "TissueForge_c_private.h"

#include <state/tfSpecies.h>
#include <state/tfSpeciesList.h>
#include <state/tfSpeciesValue.h>
#include <state/tfStateVector.h>
#include <tfParticleList.h>


using namespace TissueForge;


namespace TissueForge { 


    state::Species *castC(struct tfStateSpeciesHandle *handle) {
        return castC<state::Species, tfStateSpeciesHandle>(handle);
    }

    state::SpeciesList *castC(struct tfStateSpeciesListHandle *handle) {
        return castC<state::SpeciesList, tfStateSpeciesListHandle>(handle);
    }

    state::SpeciesValue *castC(struct tfStateSpeciesValueHandle *handle) {
        return castC<state::SpeciesValue, tfStateSpeciesValueHandle>(handle);
    }

}

#define TFC_SPECIES_GET(handle, varname) \
    state::Species *varname = TissueForge::castC<state::Species, tfStateSpeciesHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_SPECIESLIST_GET(handle, varname) \
    state::SpeciesList *varname = TissueForge::castC<state::SpeciesList, tfStateSpeciesListHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_SPECIESVALUE_GET(handle, varname) \
    state::SpeciesValue *varname = TissueForge::castC<state::SpeciesValue, tfStateSpeciesValueHandle>(handle); \
    TFC_PTRCHECK(varname);


////////////////////
// state::Species //
////////////////////


HRESULT tfStateSpecies_init(struct tfStateSpeciesHandle *handle) {
    TFC_PTRCHECK(handle);
    state::Species *species = new state::Species();
    handle->tfObj = (void*)species;
    return S_OK;
}

HRESULT tfStateSpecies_initS(struct tfStateSpeciesHandle *handle, const char *s) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(s);
    state::Species *species = new state::Species(s);
    TFC_PTRCHECK(species);
    handle->tfObj = (void*)species;
    return S_OK;
}

HRESULT tfStateSpecies_copy(struct tfStateSpeciesHandle *source, struct tfStateSpeciesHandle *destination) {
    TFC_SPECIES_GET(source, srcSpecies);
    TFC_PTRCHECK(destination);
    state::Species *dstSpecies = new state::Species(*srcSpecies);
    TFC_PTRCHECK(dstSpecies);
    destination->tfObj = (void*)dstSpecies;
    return S_OK;
}

HRESULT tfStateSpecies_destroy(struct tfStateSpeciesHandle *handle) {
    return TissueForge::capi::destroyHandle<state::Species, tfStateSpeciesHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfStateSpecies_getId(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIES_GET(handle, species);
    return TissueForge::capi::str2Char(species->getId(), str, numChars);
}

HRESULT tfStateSpecies_setId(struct tfStateSpeciesHandle *handle, const char *sid) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(sid);
    return species->setId(sid);
}

HRESULT tfStateSpecies_getName(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIES_GET(handle, species);
    return TissueForge::capi::str2Char(species->getName(), str, numChars);
}

HRESULT tfStateSpecies_setName(struct tfStateSpeciesHandle *handle, const char *name) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(name);
    return species->setName(name);
}

HRESULT tfStateSpecies_getSpeciesType(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIES_GET(handle, species);
    return TissueForge::capi::str2Char(species->getSpeciesType(), str, numChars);
}

HRESULT tfStateSpecies_setSpeciesType(struct tfStateSpeciesHandle *handle, const char *sid) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(sid);
    return species->setSpeciesType(sid);
}

HRESULT tfStateSpecies_getCompartment(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIES_GET(handle, species);
    return TissueForge::capi::str2Char(species->getCompartment(), str, numChars);
}

HRESULT tfStateSpecies_setCompartment(struct tfStateSpeciesHandle *handle, const char *sid) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(sid);
    return species->setCompartment(sid);
}

HRESULT tfStateSpecies_getInitialAmount(struct tfStateSpeciesHandle *handle, tfFloatP_t *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->getInitialAmount();
    return S_OK;
}

HRESULT tfStateSpecies_setInitialAmount(struct tfStateSpeciesHandle *handle, tfFloatP_t value) {
    TFC_SPECIES_GET(handle, species);
    return species->setInitialAmount(value);
}

HRESULT tfStateSpecies_getInitialConcentration(struct tfStateSpeciesHandle *handle, tfFloatP_t *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->getInitialConcentration();
    return S_OK;
}

HRESULT tfStateSpecies_setInitialConcentration(struct tfStateSpeciesHandle *handle, tfFloatP_t value) {
    TFC_SPECIES_GET(handle, species);
    return species->setInitialConcentration(value);
}

HRESULT tfStateSpecies_getSubstanceUnits(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIES_GET(handle, species);
    return TissueForge::capi::str2Char(species->getSubstanceUnits(), str, numChars);
}

HRESULT tfStateSpecies_setSubstanceUnits(struct tfStateSpeciesHandle *handle, const char *sid) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(sid);
    return species->setSubstanceUnits(sid);
}

HRESULT tfStateSpecies_getSpatialSizeUnits(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIES_GET(handle, species);
    return TissueForge::capi::str2Char(species->getSpatialSizeUnits(), str, numChars);
}

HRESULT tfStateSpecies_setSpatialSizeUnits(struct tfStateSpeciesHandle *handle, const char *sid) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(sid);
    return species->setSpatialSizeUnits(sid);
}

HRESULT tfStateSpecies_getUnits(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIES_GET(handle, species);
    return TissueForge::capi::str2Char(species->getUnits(), str, numChars);
}

HRESULT tfStateSpecies_setUnits(struct tfStateSpeciesHandle *handle, const char *sname) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(sname);
    return species->setUnits(sname);
}

HRESULT tfStateSpecies_getHasOnlySubstanceUnits(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->getHasOnlySubstanceUnits();
    return S_OK;
}

HRESULT tfStateSpecies_setHasOnlySubstanceUnits(struct tfStateSpeciesHandle *handle, bool value) {
    TFC_SPECIES_GET(handle, species);
    return species->setHasOnlySubstanceUnits(value);
}

HRESULT tfStateSpecies_getBoundaryCondition(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->getBoundaryCondition();
    return S_OK;
}

HRESULT tfStateSpecies_setBoundaryCondition(struct tfStateSpeciesHandle *handle, bool value) {
    TFC_SPECIES_GET(handle, species);
    return species->setBoundaryCondition(value);
}

HRESULT tfStateSpecies_getCharge(struct tfStateSpeciesHandle *handle, int *charge) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(charge);
    *charge = species->getCharge();
    return S_OK;
}

HRESULT tfStateSpecies_setCharge(struct tfStateSpeciesHandle *handle, int value) {
    TFC_SPECIES_GET(handle, species);
    return species->setCharge(value);
}

HRESULT tfStateSpecies_getConstant(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->getConstant();
    return S_OK;
}

HRESULT tfStateSpecies_setConstant(struct tfStateSpeciesHandle *handle, int value) {
    TFC_SPECIES_GET(handle, species);
    return species->setConstant(value);
}

HRESULT tfStateSpecies_getConversionFactor(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIES_GET(handle, species);
    return TissueForge::capi::str2Char(species->getConversionFactor(), str, numChars);
}

HRESULT tfStateSpecies_setConversionFactor(struct tfStateSpeciesHandle *handle, const char *sid) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(sid);
    return species->setConversionFactor(sid);
}

HRESULT tfStateSpecies_isSetId(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetId();
    return S_OK;
}

HRESULT tfStateSpecies_isSetName(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetName();
    return S_OK;
}

HRESULT tfStateSpecies_isSetSpeciesType(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetSpeciesType();
    return S_OK;
}

HRESULT tfStateSpecies_isSetCompartment(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetCompartment();
    return S_OK;
}

HRESULT tfStateSpecies_isSetInitialAmount(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetInitialAmount();
    return S_OK;
}

HRESULT tfStateSpecies_isSetInitialConcentration(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetInitialConcentration();
    return S_OK;
}

HRESULT tfStateSpecies_isSetSubstanceUnits(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetSubstanceUnits();
    return S_OK;
}

HRESULT tfStateSpecies_isSetSpatialSizeUnits(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetSpatialSizeUnits();
    return S_OK;
}

HRESULT tfStateSpecies_isSetUnits(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetUnits();
    return S_OK;
}

HRESULT tfStateSpecies_isSetCharge(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetCharge();
    return S_OK;
}

HRESULT tfStateSpecies_isSetConversionFactor(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetConversionFactor();
    return S_OK;
}

HRESULT tfStateSpecies_isSetConstant(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetConstant();
    return S_OK;
}

HRESULT tfStateSpecies_isSetBoundaryCondition(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetBoundaryCondition();
    return S_OK;
}

HRESULT tfStateSpecies_isSetHasOnlySubstanceUnits(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->isSetHasOnlySubstanceUnits();
    return S_OK;
}

HRESULT tfStateSpecies_unsetId(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    species->unsetId();
    return S_OK;
}

HRESULT tfStateSpecies_unsetName(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetName();
}

HRESULT tfStateSpecies_unsetConstant(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetConstant();
}

HRESULT tfStateSpecies_unsetSpeciesType(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetSpeciesType();
}

HRESULT tfStateSpecies_unsetInitialAmount(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetInitialAmount();
}

HRESULT tfStateSpecies_unsetInitialConcentration(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetInitialConcentration();
}

HRESULT tfStateSpecies_unsetSubstanceUnits(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetSubstanceUnits();
}

HRESULT tfStateSpecies_unsetSpatialSizeUnits(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetSpatialSizeUnits();
}

HRESULT tfStateSpecies_unsetUnits(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetUnits();
}

HRESULT tfStateSpecies_unsetCharge(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetCharge();
}

HRESULT tfStateSpecies_unsetConversionFactor(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetConversionFactor();
}

HRESULT tfStateSpecies_unsetCompartment(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetCompartment();
}

HRESULT tfStateSpecies_unsetBoundaryCondition(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetBoundaryCondition();
}

HRESULT tfStateSpecies_unsetHasOnlySubstanceUnits(struct tfStateSpeciesHandle *handle) {
    TFC_SPECIES_GET(handle, species);
    return species->unsetHasOnlySubstanceUnits();
}

HRESULT tfStateSpecies_hasRequiredAttributes(struct tfStateSpeciesHandle *handle, bool *value) {
    TFC_SPECIES_GET(handle, species);
    TFC_PTRCHECK(value);
    *value = species->hasRequiredAttributes();
    return S_OK;
}

HRESULT tfStateSpecies_toString(struct tfStateSpeciesHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIES_GET(handle, species);
    return TissueForge::capi::str2Char(species->toString(), str, numChars);
}

HRESULT tfStateSpecies_fromString(struct tfStateSpeciesHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    state::Species *species = state::Species::fromString(str);
    TFC_PTRCHECK(species);
    handle->tfObj = (void*)species;
    return S_OK;
}


////////////////////////
// state::SpeciesList //
////////////////////////


HRESULT tfStateSpeciesList_init(struct tfStateSpeciesListHandle *handle) {
    TFC_PTRCHECK(handle);
    state::SpeciesList *slist = new state::SpeciesList();
    handle->tfObj = (void*)slist;
    return S_OK;
}

HRESULT tfStateSpeciesList_destroy(struct tfStateSpeciesListHandle *handle) {
    return TissueForge::capi::destroyHandle<state::SpeciesList, tfStateSpeciesListHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfStateSpeciesList_getStr(struct tfStateSpeciesListHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIESLIST_GET(handle, slist);
    return TissueForge::capi::str2Char(slist->str(), str, numChars);
}

HRESULT tfStateSpeciesList_indexOf(struct tfStateSpeciesListHandle *handle, const char *s, unsigned int *i) {
    TFC_SPECIESLIST_GET(handle, slist);
    TFC_PTRCHECK(s);
    TFC_PTRCHECK(i);
    int _i = slist->index_of(s);
    if(_i < 0) 
        return E_FAIL;
    *i = _i;
    return S_OK;
}

HRESULT tfStateSpeciesList_getSize(struct tfStateSpeciesListHandle *handle, unsigned int *size) {
    TFC_SPECIESLIST_GET(handle, slist);
    TFC_PTRCHECK(size);
    *size = slist->size();
    return S_OK;
}

HRESULT tfStateSpeciesList_getItem(struct tfStateSpeciesListHandle *handle, unsigned int index, struct tfStateSpeciesHandle *species) {
    TFC_SPECIESLIST_GET(handle, slist);
    TFC_PTRCHECK(species);
    state::Species *_species = slist->item(index);
    TFC_PTRCHECK(_species);
    species->tfObj = (void*)_species;
    return S_OK;
}

HRESULT tfStateSpeciesList_getItemS(struct tfStateSpeciesListHandle *handle, const char *s, struct tfStateSpeciesHandle *species) {
    TFC_SPECIESLIST_GET(handle, slist);
    TFC_PTRCHECK(species);
    state::Species *_species = slist->item(s);
    TFC_PTRCHECK(_species);
    species->tfObj = (void*)_species;
    return S_OK;
}

HRESULT tfStateSpeciesList_insert(struct tfStateSpeciesListHandle *handle, struct tfStateSpeciesHandle *species) {
    TFC_SPECIESLIST_GET(handle, slist);
    TFC_SPECIES_GET(species, _species);
    return slist->insert(_species);
}

HRESULT tfStateSpeciesList_insertS(struct tfStateSpeciesListHandle *handle, const char *s) {
    TFC_SPECIESLIST_GET(handle, slist);
    TFC_PTRCHECK(s);
    return slist->insert(s);
}

HRESULT tfStateSpeciesList_toString(struct tfStateSpeciesListHandle *handle, char **str, unsigned int *numChars) {
    TFC_SPECIESLIST_GET(handle, slist);
    return TissueForge::capi::str2Char(slist->toString(), str, numChars);
}

HRESULT tfStateSpeciesList_fromString(struct tfStateSpeciesListHandle *handle, const char *str) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(str);
    state::SpeciesList *slist = state::SpeciesList::fromString(str);
    TFC_PTRCHECK(slist);
    handle->tfObj = (void*)slist;
    return S_OK;
}


/////////////////////////
// state::SpeciesValue //
/////////////////////////


HRESULT tfStateSpeciesValue_init(struct tfStateSpeciesValueHandle *handle, tfFloatP_t value, struct tfStateStateVectorHandle *state_vector, unsigned int index) {
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(state_vector); TFC_PTRCHECK(state_vector->tfObj);
    state::SpeciesValue *sval = new state::SpeciesValue(value, (state::StateVector*)state_vector->tfObj, index);
    handle->tfObj = (void*)sval;
    return S_OK;
}

HRESULT tfStateSpeciesValue_getValue(struct tfStateSpeciesValueHandle *handle, tfFloatP_t *value) {
    TFC_SPECIESVALUE_GET(handle, sval);
    TFC_PTRCHECK(value);
    *value = sval->value;
    return S_OK;
}

HRESULT tfStateSpeciesValue_setValue(struct tfStateSpeciesValueHandle *handle, tfFloatP_t value) {
    TFC_SPECIESVALUE_GET(handle, sval);
    sval->value = value;
    return S_OK;
}

HRESULT tfStateSpeciesValue_getStateVector(struct tfStateSpeciesValueHandle *handle, struct tfStateStateVectorHandle *state_vector) {
    TFC_SPECIESVALUE_GET(handle, sval);
    TFC_PTRCHECK(state_vector);
    TFC_PTRCHECK(sval->state_vector);
    state_vector->tfObj = (void*)sval->state_vector;
    return S_OK;
}

HRESULT tfStateSpeciesValue_getIndex(struct tfStateSpeciesValueHandle *handle, unsigned int *index) {
    TFC_SPECIESVALUE_GET(handle, sval);
    TFC_PTRCHECK(index);
    *index = sval->index;
    return S_OK;
}

HRESULT tfStateSpeciesValue_getSpecies(struct tfStateSpeciesValueHandle *handle, struct tfStateSpeciesHandle *species) {
    TFC_SPECIESVALUE_GET(handle, sval);
    TFC_PTRCHECK(species);
    state::Species *_species = sval->species();
    TFC_PTRCHECK(_species);
    species->tfObj = (void*)_species;
    return S_OK;
}

HRESULT tfStateSpeciesValue_getBoundaryCondition(struct tfStateSpeciesValueHandle *handle, bool *value) {
    TFC_SPECIESVALUE_GET(handle, sval);
    TFC_PTRCHECK(value);
    *value = sval->getBoundaryCondition();
    return S_OK;
}

HRESULT tfStateSpeciesValue_setBoundaryCondition(struct tfStateSpeciesValueHandle *handle, bool value) {
    TFC_SPECIESVALUE_GET(handle, sval);
    return sval->setBoundaryCondition(value);
}

HRESULT tfStateSpeciesValue_getInitialAmount(struct tfStateSpeciesValueHandle *handle, tfFloatP_t *value) {
    TFC_SPECIESVALUE_GET(handle, sval);
    TFC_PTRCHECK(value);
    *value = sval->getInitialAmount();
    return S_OK;
}

HRESULT tfStateSpeciesValue_setInitialAmount(struct tfStateSpeciesValueHandle *handle, tfFloatP_t value) {
    TFC_SPECIESVALUE_GET(handle, sval);
    return sval->setInitialAmount(value);
}

HRESULT tfStateSpeciesValue_getInitialConcentration(struct tfStateSpeciesValueHandle *handle, tfFloatP_t *value) {
    TFC_SPECIESVALUE_GET(handle, sval);
    TFC_PTRCHECK(value);
    *value = sval->getInitialConcentration();
    return S_OK;
}

HRESULT tfStateSpeciesValue_setInitialConcentration(struct tfStateSpeciesValueHandle *handle, tfFloatP_t value) {
    TFC_SPECIESVALUE_GET(handle, sval);
    return sval->setInitialConcentration(value);
}

HRESULT tfStateSpeciesValue_getConstant(struct tfStateSpeciesValueHandle *handle, bool *value) {
    TFC_SPECIESVALUE_GET(handle, sval);
    TFC_PTRCHECK(value);
    *value = sval->getConstant();
    return S_OK;
}

HRESULT tfStateSpeciesValue_setConstant(struct tfStateSpeciesValueHandle *handle, int value) {
    TFC_SPECIESVALUE_GET(handle, sval);
    return sval->setConstant(value);
}

HRESULT tfStateSpeciesValue_secreteL(struct tfStateSpeciesValueHandle *handle, tfFloatP_t amount, struct tfParticleListHandle *to, tfFloatP_t *secreted) {
    TFC_SPECIESVALUE_GET(handle, sval);
    TFC_PTRCHECK(to); TFC_PTRCHECK(to->tfObj);
    TFC_PTRCHECK(secreted);
    *secreted = sval->secrete(amount, *(ParticleList*)to->tfObj);
    return S_OK;
}

HRESULT tfStateSpeciesValue_secreteD(struct tfStateSpeciesValueHandle *handle, tfFloatP_t amount, tfFloatP_t distance, tfFloatP_t *secreted) {
    TFC_SPECIESVALUE_GET(handle, sval);
    TFC_PTRCHECK(secreted);
    *secreted = sval->secrete(amount, distance);
    return S_OK;
}
