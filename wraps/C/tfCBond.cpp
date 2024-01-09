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

#include "tfCBond.h"

#include "TissueForge_c_private.h"

#include <tf_fptype.h>
#include <io/tf_io.h>
#include <tfBond.h>
#include <tfAngle.h>
#include <tfDihedral.h>
#include <tfPotential.h>
#include <rendering/tfStyle.h>


using namespace TissueForge;


namespace TissueForge { 
    

    BondHandle *castC(struct tfBondHandleHandle *handle) {
        return castC<BondHandle, tfBondHandleHandle>(handle);
    }

    AngleHandle *castC(struct tfAngleHandleHandle *handle) {
        return castC<AngleHandle, tfAngleHandleHandle>(handle);
    }

    DihedralHandle *castC(struct tfDihedralHandleHandle *handle) {
        return castC<DihedralHandle, tfDihedralHandleHandle>(handle);
    }

}

#define TFC_BONDHANDLE_GET(handle, varname) \
    BondHandle *varname = TissueForge::castC<BondHandle, tfBondHandleHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_ANGLEHANDLE_GET(handle, varname) \
    AngleHandle *varname = TissueForge::castC<AngleHandle, tfAngleHandleHandle>(handle); \
    TFC_PTRCHECK(varname);

#define TFC_DIHEDRALHANDLE_GET(handle, varname) \
    DihedralHandle *varname = TissueForge::castC<DihedralHandle, tfDihedralHandleHandle>(handle); \
    TFC_PTRCHECK(varname);


//////////////
// Generics //
//////////////


namespace TissueForge { 
    

    namespace capi {


        template <typename O, typename H> 
        HRESULT getBondId(H *handle, int *id) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(id);
            *id = bhandle->getId();
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT getBondStr(H *handle, char **str, unsigned int *numChars) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            return TissueForge::capi::str2Char(bhandle->str(), str, numChars);
        }

        template <typename O, typename H> 
        HRESULT bondCheck(H *handle, bool *flag) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(flag);
            *flag = bhandle->check();
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT bondDecays(H *handle, bool *flag) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(flag);
            *flag = bhandle->decays();
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT getBondEnergy(H *handle, tfFloatP_t *value) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(value);
            *value = bhandle->getEnergy();
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT getBondPotential(H *handle, struct tfPotentialHandle *potential) {
            TFC_PTRCHECK(potential);
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            Potential *_potential = bhandle->getPotential();
            TFC_PTRCHECK(_potential);
            potential->tfObj = (void*)_potential;
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT getBondDissociationEnergy(H *handle, tfFloatP_t *value) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(value);
            *value = bhandle->getDissociationEnergy();
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT setBondDissociationEnergy(H *handle, const tfFloatP_t &value) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            bhandle->setDissociationEnergy(value);
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT getBondHalfLife(H *handle, tfFloatP_t *value) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(value);
            *value = bhandle->getHalfLife();
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT setBondHalfLife(H *handle, const tfFloatP_t &value) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            bhandle->setHalfLife(value);
            return S_OK;
        }

        template <typename O, typename H, typename B> 
        HRESULT setBondActive(H *handle, const bool &flag, const unsigned int &activeFlag) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            B *b = bhandle->get();
            b->flags |= activeFlag;
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT getBondStyle(H *handle, struct tfRenderingStyleHandle *style) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(style);
            style->tfObj = (void*)bhandle->getStyle();
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT setBondStyle(H *handle, struct tfRenderingStyleHandle *style) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(style);
            TFC_PTRCHECK(style->tfObj);
            bhandle->setStyle((rendering::Style*)style->tfObj);
            return S_OK;
        }

        template <typename H> 
        HRESULT getBondStyleDef(struct tfRenderingStyleHandle *style) {
            TFC_PTRCHECK(style);
            style->tfObj = (void*)H::styleDef();
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT getBondAge(H *handle, tfFloatP_t *value) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(value);
            *value = bhandle->getAge();
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT getBondPartList(H *handle, tfParticleListHandle *plist) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(plist);
            plist->tfObj = (void*)(new ParticleList(bhandle->getPartList()));
            return S_OK;
        }

        template <typename O, typename H, typename D> 
        HRESULT bondHas(H *handle, D data, bool *result) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            TFC_PTRCHECK(result);
            *result = bhandle->has(data);
            return S_OK;
        }

        template <typename O, typename H, typename B> 
        HRESULT bondToString(H *handle, char **str, unsigned int *numChars) {
            O *bhandle = castC<O, H>(handle);
            TFC_PTRCHECK(bhandle);
            B *b = bhandle->get();
            TFC_PTRCHECK(b);
            return TissueForge::capi::str2Char(b->toString(), str, numChars);
        }

        template <typename O, typename H, typename B> 
        HRESULT bondFromString(H *handle, const char *str) {
            TFC_PTRCHECK(str);
            B *b = B::fromString(str);
            TFC_PTRCHECK(b);
            O *bhandle = new O(b->id);
            handle->tfObj = (void*)bhandle;
            return S_OK;
        }

        template <typename O, typename H> 
        HRESULT getAllBonds(H **handles, unsigned int *numBonds) {
            TFC_PTRCHECK(handles);
            TFC_PTRCHECK(numBonds);

            auto _items = O::items();
            *numBonds = _items.size();
            if(*numBonds == 0) 
                return S_OK;

            H *_handles = (H*)malloc(*numBonds * sizeof(H));
            if(!_handles) 
                return E_OUTOFMEMORY;
            for(unsigned int i = 0; i < *numBonds; i++) 
                _handles[i].tfObj = (void*)(new O(_items[i]));
            *handles = _handles;
            return S_OK;
        }

        HRESULT passBondIdsForParticle(const std::vector<int32_t> &items, unsigned int **bids, unsigned int *numIds) {
            TFC_PTRCHECK(bids);
            TFC_PTRCHECK(numIds);

            *numIds = items.size();
            if(*numIds == 0) 
                return S_OK;

            unsigned int *_bids = (unsigned int*)malloc(*numIds * sizeof(unsigned int));
            if(!_bids) 
                return E_OUTOFMEMORY;
            for(unsigned int i = 0; i < *numIds; i++) 
                _bids[i] = items[i];
            *bids = _bids;
            return S_OK;
        }

}}


////////////////
// BondHandle //
////////////////


HRESULT tfBondHandle_init(struct tfBondHandleHandle *handle, unsigned int id) {
    TFC_PTRCHECK(handle);
    BondHandle *bhandle = new BondHandle(id);
    handle->tfObj = (void*)bhandle;
    return S_OK;
}

HRESULT tfBondHandle_create(struct tfBondHandleHandle *handle, 
                             struct tfPotentialHandle *potential,
                             struct tfParticleHandleHandle *i, 
                             struct tfParticleHandleHandle *j) 
{
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(potential); TFC_PTRCHECK(potential->tfObj);
    TFC_PTRCHECK(i); TFC_PTRCHECK(i->tfObj);
    TFC_PTRCHECK(j); TFC_PTRCHECK(j->tfObj);
    BondHandle *bhandle = Bond::create((Potential*)potential->tfObj, 
                                           (ParticleHandle*)i->tfObj, 
                                           (ParticleHandle*)j->tfObj);
    TFC_PTRCHECK(bhandle);
    handle->tfObj = (void*)bhandle;
    return S_OK;
}

HRESULT tfBondHandle_getId(struct tfBondHandleHandle *handle, int *id) {
    return TissueForge::capi::getBondId<BondHandle, tfBondHandleHandle>(handle, id);
}

HRESULT tfBondHandle_str(struct tfBondHandleHandle *handle, char **str, unsigned int *numChars) {
    return TissueForge::capi::getBondStr<BondHandle, tfBondHandleHandle>(handle, str, numChars);
}

HRESULT tfBondHandle_check(struct tfBondHandleHandle *handle, bool *flag) {
    return TissueForge::capi::bondCheck<BondHandle, tfBondHandleHandle>(handle, flag);
}

HRESULT tfBondHandle_destroy(struct tfBondHandleHandle *handle) {
    return TissueForge::capi::destroyHandle<BondHandle, tfBondHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfBondHandle_decays(struct tfBondHandleHandle *handle, bool *flag) {
    return TissueForge::capi::bondDecays<BondHandle, tfBondHandleHandle>(handle, flag);
}

HRESULT tfBondHandle_hasPartId(struct tfBondHandleHandle *handle, int pid, bool *result) {
    return TissueForge::capi::bondHas<BondHandle, tfBondHandleHandle, int>(handle, pid, result);
}

HRESULT tfBondHandle_hasPart(struct tfBondHandleHandle *handle, struct tfParticleHandleHandle *part, bool *result) {
    TFC_PTRCHECK(part);
    return TissueForge::capi::bondHas<BondHandle, tfBondHandleHandle, ParticleHandle*>(handle, (ParticleHandle*)part->tfObj, result);
}

HRESULT tfBondHandle_getEnergy(struct tfBondHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondEnergy<BondHandle, tfBondHandleHandle>(handle, value);
}

HRESULT tfBondHandle_getParts(struct tfBondHandleHandle *handle, int *parti, int *partj) {
    TFC_BONDHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(parti);
    TFC_PTRCHECK(partj);
    auto pids = bhandle->getParts();
    *parti = pids[0];
    *partj = pids[1];
    return S_OK;
}

HRESULT tfBondHandle_getPartList(struct tfBondHandleHandle *handle, struct tfParticleListHandle *plist) {
    return TissueForge::capi::getBondPartList<BondHandle, tfBondHandleHandle>(handle, plist);
}

HRESULT tfBondHandle_getPotential(struct tfBondHandleHandle *handle, struct tfPotentialHandle *potential) {
    return TissueForge::capi::getBondPotential<BondHandle, tfBondHandleHandle>(handle, potential);
}

HRESULT tfBondHandle_getDissociationEnergy(struct tfBondHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondDissociationEnergy<BondHandle, tfBondHandleHandle>(handle, value);
}

HRESULT tfBondHandle_setDissociationEnergy(struct tfBondHandleHandle *handle, tfFloatP_t value) {
    return TissueForge::capi::setBondDissociationEnergy<BondHandle, tfBondHandleHandle>(handle, value);
}

HRESULT tfBondHandle_getHalfLife(struct tfBondHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondHalfLife<BondHandle, tfBondHandleHandle>(handle, value);
}

HRESULT tfBondHandle_setHalfLife(struct tfBondHandleHandle *handle, tfFloatP_t value) {
    return TissueForge::capi::setBondHalfLife<BondHandle, tfBondHandleHandle>(handle, value);
}

HRESULT tfBondHandle_setActive(struct tfBondHandleHandle *handle, bool flag) {
    return TissueForge::capi::setBondActive<BondHandle, tfBondHandleHandle, Bond>(handle, flag, BOND_ACTIVE);
}

HRESULT tfBondHandle_getStyle(struct tfBondHandleHandle *handle, struct tfRenderingStyleHandle *style) {
    return TissueForge::capi::getBondStyle<BondHandle, tfBondHandleHandle>(handle, style);
}

HRESULT tfBondHandle_setStyle(struct tfBondHandleHandle *handle, struct tfRenderingStyleHandle *style) {
    return TissueForge::capi::setBondStyle<BondHandle, tfBondHandleHandle>(handle, style);
}

HRESULT tfBondHandle_getStyleDef(struct tfRenderingStyleHandle *style) {
    return TissueForge::capi::getBondStyleDef<Bond>(style);
}

HRESULT tfBondHandle_getAge(struct tfBondHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondAge<BondHandle, tfBondHandleHandle>(handle, value);
}

HRESULT tfBondHandle_toString(struct tfBondHandleHandle *handle, char **str, unsigned int *numChars) {
    return TissueForge::capi::bondToString<BondHandle, tfBondHandleHandle, Bond>(handle, str, numChars);
}

HRESULT tfBondHandle_fromString(struct tfBondHandleHandle *handle, const char *str) {
    return TissueForge::capi::bondFromString<BondHandle, tfBondHandleHandle, Bond>(handle, str);
}

HRESULT tfBondHandle_lt(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_lt<BondHandle, tfBondHandleHandle>(lhs, rhs, result);
}

HRESULT tfBondHandle_gt(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_gt<BondHandle, tfBondHandleHandle>(lhs, rhs, result);
}

HRESULT tfBondHandle_le(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_le<BondHandle, tfBondHandleHandle>(lhs, rhs, result);
}

HRESULT tfBondHandle_ge(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_ge<BondHandle, tfBondHandleHandle>(lhs, rhs, result);
}

HRESULT tfBondHandle_eq(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_eq<BondHandle, tfBondHandleHandle>(lhs, rhs, result);
}

HRESULT tfBondHandle_ne(struct tfBondHandleHandle *lhs, struct tfBondHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_ne<BondHandle, tfBondHandleHandle>(lhs, rhs, result);
}


/////////////////
// AngleHandle //
/////////////////


HRESULT tfAngleHandle_init(struct tfAngleHandleHandle *handle, unsigned int id) {
    TFC_PTRCHECK(handle);
    AngleHandle *bhandle = new AngleHandle(id);
    handle->tfObj = (void*)bhandle;
    return S_OK;
}

HRESULT tfAngleHandle_create(struct tfAngleHandleHandle *handle, 
                              struct tfPotentialHandle *potential,
                              struct tfParticleHandleHandle *i, 
                              struct tfParticleHandleHandle *j, 
                              struct tfParticleHandleHandle *k) 
{
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(potential); TFC_PTRCHECK(potential->tfObj);
    TFC_PTRCHECK(i); TFC_PTRCHECK(i->tfObj);
    TFC_PTRCHECK(j); TFC_PTRCHECK(j->tfObj);
    AngleHandle *bhandle = Angle::create((Potential*)potential->tfObj, 
                                             (ParticleHandle*)i->tfObj, 
                                             (ParticleHandle*)j->tfObj, 
                                             (ParticleHandle*)k->tfObj);
    TFC_PTRCHECK(bhandle);
    handle->tfObj = (void*)bhandle;
    return S_OK;
}

HRESULT tfAngleHandle_getId(struct tfAngleHandleHandle *handle, int *id) {
    return TissueForge::capi::getBondId<AngleHandle, tfAngleHandleHandle>(handle, id);
}

HRESULT tfAngleHandle_getStr(struct tfAngleHandleHandle *handle, char **str, unsigned int *numChars) {
    return TissueForge::capi::getBondStr<AngleHandle, tfAngleHandleHandle>(handle, str, numChars);
}

HRESULT tfAngleHandle_str(struct tfAngleHandleHandle *handle, char **str, unsigned int *numChars) {
    return TissueForge::capi::getBondStr<AngleHandle, tfAngleHandleHandle>(handle, str, numChars);
}

HRESULT tfAngleHandle_check(struct tfAngleHandleHandle *handle, bool *flag) {
    return TissueForge::capi::bondCheck<AngleHandle, tfAngleHandleHandle>(handle, flag);
}

HRESULT tfAngleHandle_destroy(struct tfAngleHandleHandle *handle) {
    return TissueForge::capi::destroyHandle<AngleHandle, tfAngleHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfAngleHandle_decays(struct tfAngleHandleHandle *handle, bool *flag) {
    return TissueForge::capi::bondDecays<AngleHandle, tfAngleHandleHandle>(handle, flag);
}

HRESULT tfAngleHandle_hasPartId(struct tfAngleHandleHandle *handle, int pid, bool *result) {
    return TissueForge::capi::bondHas<AngleHandle, tfAngleHandleHandle, int>(handle, pid, result);
}

HRESULT tfAngleHandle_hasPart(struct tfAngleHandleHandle *handle, struct tfParticleHandleHandle *part, bool *result) {
    TFC_PTRCHECK(part);
    return TissueForge::capi::bondHas<AngleHandle, tfAngleHandleHandle, ParticleHandle*>(handle, (ParticleHandle*)part->tfObj, result);
}

HRESULT tfAngleHandle_getEnergy(struct tfAngleHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondEnergy<AngleHandle, tfAngleHandleHandle>(handle, value);
}

HRESULT tfAngleHandle_getParts(struct tfAngleHandleHandle *handle, int *parti, int *partj, int *partk) {
    TFC_ANGLEHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(parti);
    TFC_PTRCHECK(partj);
    TFC_PTRCHECK(partk);
    auto pids = bhandle->getParts();
    *parti = pids[0];
    *partj = pids[1];
    *partk = pids[2];
    return S_OK;
}

HRESULT tfAngleHandle_getPartList(struct tfAngleHandleHandle *handle, struct tfParticleListHandle *plist) {
    return TissueForge::capi::getBondPartList<AngleHandle, tfAngleHandleHandle>(handle, plist);
}

HRESULT tfAngleHandle_getPotential(struct tfAngleHandleHandle *handle, struct tfPotentialHandle *potential) {
    return TissueForge::capi::getBondPotential<AngleHandle, tfAngleHandleHandle>(handle, potential);
}

HRESULT tfAngleHandle_getDissociationEnergy(struct tfAngleHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondDissociationEnergy<AngleHandle, tfAngleHandleHandle>(handle, value);
}

HRESULT tfAngleHandle_setDissociationEnergy(struct tfAngleHandleHandle *handle, tfFloatP_t value) {
    return TissueForge::capi::setBondDissociationEnergy<AngleHandle, tfAngleHandleHandle>(handle, value);
}

HRESULT tfAngleHandle_getHalfLife(struct tfAngleHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondHalfLife<AngleHandle, tfAngleHandleHandle>(handle, value);
}

HRESULT tfAngleHandle_setHalfLife(struct tfAngleHandleHandle *handle, tfFloatP_t value) {
    return TissueForge::capi::setBondHalfLife<AngleHandle, tfAngleHandleHandle>(handle, value);
}

HRESULT tfAngleHandle_setActive(struct tfAngleHandleHandle *handle, bool flag) {
    return TissueForge::capi::setBondActive<AngleHandle, tfAngleHandleHandle, Angle>(handle, flag, ANGLE_ACTIVE);
}

HRESULT tfAngleHandle_getStyle(struct tfAngleHandleHandle *handle, struct tfRenderingStyleHandle *style) {
    return TissueForge::capi::getBondStyle<AngleHandle, tfAngleHandleHandle>(handle, style);
}

HRESULT tfAngleHandle_setStyle(struct tfAngleHandleHandle *handle, struct tfRenderingStyleHandle *style) {
    return TissueForge::capi::setBondStyle<AngleHandle, tfAngleHandleHandle>(handle, style);
}

HRESULT tfAngleHandle_getStyleDef(struct tfRenderingStyleHandle *style) {
    return TissueForge::capi::getBondStyleDef<Angle>(style);
}

HRESULT tfAngleHandle_getAge(struct tfAngleHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondAge<AngleHandle, tfAngleHandleHandle>(handle, value);
}

HRESULT tfAngleHandle_toString(struct tfAngleHandleHandle *handle, char **str, unsigned int *numChars) {
    return TissueForge::capi::bondToString<AngleHandle, tfAngleHandleHandle, Angle>(handle, str, numChars);
}

HRESULT tfAngleHandle_fromString(struct tfAngleHandleHandle *handle, const char *str) {
    return TissueForge::capi::bondFromString<AngleHandle, tfAngleHandleHandle, Angle>(handle, str);
}

HRESULT tfAngleHandle_lt(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_lt<AngleHandle, tfAngleHandleHandle>(lhs, rhs, result);
}

HRESULT tfAngleHandle_gt(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_gt<AngleHandle, tfAngleHandleHandle>(lhs, rhs, result);
}

HRESULT tfAngleHandle_le(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_le<AngleHandle, tfAngleHandleHandle>(lhs, rhs, result);
}

HRESULT tfAngleHandle_ge(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_ge<AngleHandle, tfAngleHandleHandle>(lhs, rhs, result);
}

HRESULT tfAngleHandle_eq(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_eq<AngleHandle, tfAngleHandleHandle>(lhs, rhs, result);
}

HRESULT tfAngleHandle_ne(struct tfAngleHandleHandle *lhs, struct tfAngleHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_ne<AngleHandle, tfAngleHandleHandle>(lhs, rhs, result);
}


////////////////////
// DihedralHandle //
////////////////////


HRESULT tfDihedralHandle_init(struct tfDihedralHandleHandle *handle, unsigned int id) {
    TFC_PTRCHECK(handle);
    DihedralHandle *bhandle = new DihedralHandle(id);
    handle->tfObj = (void*)bhandle;
    return S_OK;
}

HRESULT tfDihedralHandle_create(struct tfDihedralHandleHandle *handle, 
                                 struct tfPotentialHandle *potential,
                                 struct tfParticleHandleHandle *i, 
                                 struct tfParticleHandleHandle *j, 
                                 struct tfParticleHandleHandle *k, 
                                 struct tfParticleHandleHandle *l) 
{
    TFC_PTRCHECK(handle);
    TFC_PTRCHECK(potential); TFC_PTRCHECK(potential->tfObj);
    TFC_PTRCHECK(i); TFC_PTRCHECK(i->tfObj);
    TFC_PTRCHECK(j); TFC_PTRCHECK(j->tfObj);
    TFC_PTRCHECK(k); TFC_PTRCHECK(k->tfObj);
    TFC_PTRCHECK(l); TFC_PTRCHECK(l->tfObj);
    DihedralHandle *bhandle = Dihedral::create((Potential*)potential->tfObj, 
                                                   (ParticleHandle*)i->tfObj, 
                                                   (ParticleHandle*)j->tfObj, 
                                                   (ParticleHandle*)k->tfObj, 
                                                   (ParticleHandle*)l->tfObj);
    TFC_PTRCHECK(bhandle);
    handle->tfObj = (void*)bhandle;
    return S_OK;
}

HRESULT tfDihedralHandle_getId(struct tfDihedralHandleHandle *handle, int *id) {
    return TissueForge::capi::getBondId<DihedralHandle, tfDihedralHandleHandle>(handle, id);
}

HRESULT tfDihedralHandle_getStr(struct tfDihedralHandleHandle *handle, char **str, unsigned int *numChars) {
    return TissueForge::capi::getBondStr<DihedralHandle, tfDihedralHandleHandle>(handle, str, numChars);
}

HRESULT tfDihedralHandle_str(struct tfDihedralHandleHandle *handle, char **str, unsigned int *numChars) {
    return TissueForge::capi::getBondStr<DihedralHandle, tfDihedralHandleHandle>(handle, str, numChars);
}

HRESULT tfDihedralHandle_check(struct tfDihedralHandleHandle *handle, bool *flag) {
    return TissueForge::capi::bondCheck<DihedralHandle, tfDihedralHandleHandle>(handle, flag);
}

HRESULT tfDihedralHandle_destroy(struct tfDihedralHandleHandle *handle) {
    return TissueForge::capi::destroyHandle<DihedralHandle, tfDihedralHandleHandle>(handle) ? S_OK : E_FAIL;
}

HRESULT tfDihedralHandle_decays(struct tfDihedralHandleHandle *handle, bool *flag) {
    return TissueForge::capi::bondDecays<DihedralHandle, tfDihedralHandleHandle>(handle, flag);
}

HRESULT tfDihedralHandle_hasPartId(struct tfDihedralHandleHandle *handle, int pid, bool *result) {
    return TissueForge::capi::bondHas<DihedralHandle, tfDihedralHandleHandle, int>(handle, pid, result);
}

HRESULT tfDihedralHandle_hasPart(struct tfDihedralHandleHandle *handle, struct tfParticleHandleHandle *part, bool *result) {
    TFC_PTRCHECK(part);
    return TissueForge::capi::bondHas<DihedralHandle, tfDihedralHandleHandle, ParticleHandle*>(handle, (ParticleHandle*)part->tfObj, result);
}

HRESULT tfDihedralHandle_getEnergy(struct tfDihedralHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondEnergy<DihedralHandle, tfDihedralHandleHandle>(handle, value);
}

HRESULT tfDihedralHandle_getParts(struct tfDihedralHandleHandle *handle, int *parti, int *partj, int *partk, int *partl) {
    TFC_DIHEDRALHANDLE_GET(handle, bhandle);
    TFC_PTRCHECK(parti);
    TFC_PTRCHECK(partj);
    TFC_PTRCHECK(partk);
    TFC_PTRCHECK(partl);
    auto pids = bhandle->getParts();
    *parti = pids[0];
    *partj = pids[1];
    *partk = pids[2];
    *partl = pids[3];
    return S_OK;
}

HRESULT tfDihedralHandle_getPartList(struct tfDihedralHandleHandle *handle, struct tfParticleListHandle *plist) {
    return TissueForge::capi::getBondPartList<DihedralHandle, tfDihedralHandleHandle>(handle, plist);
}

HRESULT tfDihedralHandle_getPotential(struct tfDihedralHandleHandle *handle, struct tfPotentialHandle *potential) {
    return TissueForge::capi::getBondPotential<DihedralHandle, tfDihedralHandleHandle>(handle, potential);
}

HRESULT tfDihedralHandle_getDissociationEnergy(struct tfDihedralHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondDissociationEnergy<DihedralHandle, tfDihedralHandleHandle>(handle, value);
}

HRESULT tfDihedralHandle_setDissociationEnergy(struct tfDihedralHandleHandle *handle, tfFloatP_t value) {
    return TissueForge::capi::setBondDissociationEnergy<DihedralHandle, tfDihedralHandleHandle>(handle, value);
}

HRESULT tfDihedralHandle_getHalfLife(struct tfDihedralHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondHalfLife<DihedralHandle, tfDihedralHandleHandle>(handle, value);
}

HRESULT tfDihedralHandle_setHalfLife(struct tfDihedralHandleHandle *handle, tfFloatP_t value) {
    return TissueForge::capi::setBondHalfLife<DihedralHandle, tfDihedralHandleHandle>(handle, value);
}

HRESULT tfDihedralHandle_setActive(struct tfDihedralHandleHandle *handle, bool flag) {
    return TissueForge::capi::setBondActive<DihedralHandle, tfDihedralHandleHandle, Dihedral>(handle, flag, DIHEDRAL_ACTIVE);
}

HRESULT tfDihedralHandle_getStyle(struct tfDihedralHandleHandle *handle, struct tfRenderingStyleHandle *style) {
    return TissueForge::capi::getBondStyle<DihedralHandle, tfDihedralHandleHandle>(handle, style);
}

HRESULT tfDihedralHandle_setStyle(struct tfDihedralHandleHandle *handle, struct tfRenderingStyleHandle *style) {
    return TissueForge::capi::setBondStyle<DihedralHandle, tfDihedralHandleHandle>(handle, style);
}

HRESULT tfDihedralHandle_getStyleDef(struct tfRenderingStyleHandle *style) {
    return TissueForge::capi::getBondStyleDef<Dihedral>(style);
}

HRESULT tfDihedralHandle_getAge(struct tfDihedralHandleHandle *handle, tfFloatP_t *value) {
    return TissueForge::capi::getBondAge<DihedralHandle, tfDihedralHandleHandle>(handle, value);
}

HRESULT tfDihedralHandle_toString(struct tfDihedralHandleHandle *handle, char **str, unsigned int *numChars) {
    return TissueForge::capi::bondToString<DihedralHandle, tfDihedralHandleHandle, Dihedral>(handle, str, numChars);
}

HRESULT tfDihedralHandle_fromString(struct tfDihedralHandleHandle *handle, const char *str) {
    return TissueForge::capi::bondFromString<DihedralHandle, tfDihedralHandleHandle, Dihedral>(handle, str);
}

HRESULT tfDihedralHandle_lt(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_lt<DihedralHandle, tfDihedralHandleHandle>(lhs, rhs, result);
}

HRESULT tfDihedralHandle_gt(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_gt<DihedralHandle, tfDihedralHandleHandle>(lhs, rhs, result);
}

HRESULT tfDihedralHandle_le(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_le<DihedralHandle, tfDihedralHandleHandle>(lhs, rhs, result);
}

HRESULT tfDihedralHandle_ge(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_ge<DihedralHandle, tfDihedralHandleHandle>(lhs, rhs, result);
}

HRESULT tfDihedralHandle_eq(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_eq<DihedralHandle, tfDihedralHandleHandle>(lhs, rhs, result);
}

HRESULT tfDihedralHandle_ne(struct tfDihedralHandleHandle *lhs, struct tfDihedralHandleHandle *rhs, bool *result) {
    return TissueForge::capi::obj_ne<DihedralHandle, tfDihedralHandleHandle>(lhs, rhs, result);
}

//////////////////////
// Module functions //
//////////////////////


HRESULT tfBondHandle_getAll(struct tfBondHandleHandle **handles, unsigned int *numBonds) {
    return TissueForge::capi::getAllBonds<BondHandle, tfBondHandleHandle>(handles, numBonds);
}

HRESULT tfBond_pairwise(
    struct tfPotentialHandle *pot, 
    struct tfParticleListHandle *parts, 
    tfFloatP_t cutoff, 
    struct tfParticleTypeHandle *ppairsA, 
    struct tfParticleTypeHandle *ppairsB, 
    unsigned int numTypePairs, 
    tfFloatP_t *half_life, 
    tfFloatP_t *bond_energy, 
    struct tfBondHandleHandle **bonds, 
    unsigned int *numBonds) 
{
    TFC_PTRCHECK(pot); TFC_PTRCHECK(pot->tfObj);
    TFC_PTRCHECK(parts); TFC_PTRCHECK(parts->tfObj);
    TFC_PTRCHECK(ppairsA);
    TFC_PTRCHECK(ppairsB);
    TFC_PTRCHECK(bonds);

    std::vector<std::pair<ParticleType*, ParticleType*> *> ppairs;
    tfParticleTypeHandle pta, ptb;
    for(unsigned int i = 0; i < numTypePairs; i++) {
        pta = ppairsA[i];
        ptb = ppairsB[i];
        if(!pta.tfObj || !ptb.tfObj) 
            return E_FAIL;
        ppairs.push_back(new std::pair<ParticleType*, ParticleType*>(std::make_pair((ParticleType*)pta.tfObj, (ParticleType*)ptb.tfObj)));
    }
    tfFloatP_t _half_life = half_life ? *half_life : 0.0;
    tfFloatP_t _bond_energy = bond_energy ? *bond_energy : 0.0;
    auto _items = BondHandle::pairwise((Potential*)pot->tfObj, *(ParticleList*)parts->tfObj, cutoff, &ppairs, _half_life, _bond_energy, BOND_ACTIVE);
    for(unsigned int i = 0; i < numTypePairs; i++) 
        delete ppairs[i];

    *numBonds = _items.size();
    if(*numBonds == 0) 
        return S_OK;

    tfBondHandleHandle *_bonds = (tfBondHandleHandle*)malloc(*numBonds * sizeof(tfBondHandleHandle));
    if(!_bonds) 
        return E_OUTOFMEMORY;
    for(unsigned int i = 0; i < *numBonds; i++) 
        _bonds[i].tfObj = (void*)(new BondHandle(_items[i]));
    *bonds = _bonds;
    return S_OK;
}

HRESULT tfBond_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds) {
    TFC_PTRCHECK(bids)
    return TissueForge::capi::passBondIdsForParticle(Bond_IdsForParticle(pid), bids, numIds);
}

HRESULT tfBond_destroyAll() {
    return Bond_DestroyAll();
}

HRESULT tfAngleHandle_getAll(struct tfAngleHandleHandle **handles, unsigned int *numBonds) {
    return TissueForge::capi::getAllBonds<AngleHandle, tfAngleHandleHandle>(handles, numBonds);
}

HRESULT tfAngle_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds) {
    TFC_PTRCHECK(bids);
    return TissueForge::capi::passBondIdsForParticle(Angle_IdsForParticle(pid), bids, numIds);
}

HRESULT tfAngle_destroyAll() {
    return Angle_DestroyAll();
}

HRESULT tfDihedralHandle_getAll(struct tfDihedralHandleHandle **handles, unsigned int *numBonds) {
    return TissueForge::capi::getAllBonds<DihedralHandle, tfDihedralHandleHandle>(handles, numBonds);
}

HRESULT tfDihedral_getIdsForParticle(unsigned int pid, unsigned int **bids, unsigned int *numIds) {
    TFC_PTRCHECK(bids);
    return TissueForge::capi::passBondIdsForParticle(Dihedral_IdsForParticle(pid), bids, numIds);
}

HRESULT tfDihedral_destroyAll() {
    return Dihedral_DestroyAll();
}
