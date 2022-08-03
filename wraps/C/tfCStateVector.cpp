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

#include "tfCStateVector.h"

#include "TissueForge_c_private.h"

#include "tfCSpecies.h"

#include <state/tfStateVector.h>


using namespace TissueForge;


namespace TissueForge { 


    state::StateVector *castC(struct tfStateStateVectorHandle *handle) {
        return castC<state::StateVector, tfStateStateVectorHandle>(handle);
    }

}

#define TFC_STATEVECTOR_GET(handle) \
    state::StateVector *svec = TissueForge::castC<state::StateVector, tfStateStateVectorHandle>(handle); \
    TFC_PTRCHECK(svec);


//////////////////////
// stateStateVector //
//////////////////////


HRESULT tfStateStateVector_destroy(struct tfStateStateVectorHandle *handle) {
    return TissueForge::capi::destroyHandle<state::StateVector, tfStateStateVectorHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfStateStateVector_getSize(struct tfStateStateVectorHandle *handle, unsigned int *size) {
    TFC_STATEVECTOR_GET(handle);
    TFC_PTRCHECK(size);
    *size = svec->size;
    return S_OK;
}

HRESULT tfStateStateVector_getSpecies(struct tfStateStateVectorHandle *handle, struct tfStateSpeciesListHandle *slist) {
    TFC_STATEVECTOR_GET(handle);
    TFC_PTRCHECK(slist);
    slist->tfObj = (void*)svec->species;
    return S_OK;
}

HRESULT tfStateStateVector_reset(struct tfStateStateVectorHandle *handle) {
    TFC_STATEVECTOR_GET(handle);
    svec->reset();
    return S_OK;
}

HRESULT tfStateStateVector_getStr(struct tfStateStateVectorHandle *handle, char **str, unsigned int *numChars) {
    TFC_STATEVECTOR_GET(handle);
    return TissueForge::capi::str2Char(svec->str(), str, numChars);
}

HRESULT tfStateStateVector_getItem(struct tfStateStateVectorHandle *handle, int i, tfFloatP_t *value) {
    TFC_STATEVECTOR_GET(handle);
    TFC_PTRCHECK(value);
    tfFloatP_t *_value = svec->item(i);
    TFC_PTRCHECK(_value);
    *value = *_value;
    return S_OK;
}

HRESULT tfStateStateVector_setItem(struct tfStateStateVectorHandle *handle, int i, tfFloatP_t value) {
    TFC_STATEVECTOR_GET(handle);
    svec->setItem(i, value);
    return S_OK;
}

HRESULT tfStateStateVector_toString(struct tfStateStateVectorHandle *handle, char **str, unsigned int *numChars) {
    TFC_STATEVECTOR_GET(handle);
    return TissueForge::capi::str2Char(svec->toString(), str, numChars);
}

HRESULT tfStateStateVector_fromString(struct tfStateStateVectorHandle *handle, const char *str) {
    state::StateVector *svec = state::StateVector::fromString(str);
    TFC_PTRCHECK(svec);
    handle->tfObj = (void*)svec;
    return S_OK;
}
