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

#include "tfC_bind.h"

#include "TissueForge_c_private.h"

#include <tf_bind.h>
#include <tfParticle.h>
#include <tfForce.h>
#include <tfEngine.h>


using namespace TissueForge;


#define TFC_BIND_CHECKHANDLE(h) if(!h || !h->tfObj) return E_FAIL;


//////////////////////
// Module functions //
//////////////////////


HRESULT tfBindParticles(struct tfPotentialHandle *p, struct tfParticleHandleHandle *a, struct tfParticleHandleHandle *b) {
    TFC_BIND_CHECKHANDLE(p);
    TFC_BIND_CHECKHANDLE(a);
    TFC_BIND_CHECKHANDLE(b);
    ParticleHandle *ah = (ParticleHandle*)a->tfObj;
    ParticleHandle *bh = (ParticleHandle*)b->tfObj;
    return bind::particles((Potential*)p->tfObj, ah->part(), bh->part());
}

HRESULT tfBindTypes(struct tfPotentialHandle *p, struct tfParticleTypeHandle *a, struct tfParticleTypeHandle *b, bool bound) {
    TFC_BIND_CHECKHANDLE(p);
    TFC_BIND_CHECKHANDLE(a);
    TFC_BIND_CHECKHANDLE(b);
    return bind::types((Potential*)p->tfObj, (ParticleType*)a->tfObj, (ParticleType*)b->tfObj, bound);
}

HRESULT tfBindBoundaryConditions(struct tfPotentialHandle *p, struct tfParticleTypeHandle *t) {
    TFC_BIND_CHECKHANDLE(p);
    TFC_BIND_CHECKHANDLE(t);
    return bind::boundaryConditions((Potential*)p->tfObj, (ParticleType*)t->tfObj);
}

HRESULT tfBindBoundaryCondition(struct tfPotentialHandle *p, struct tfBoundaryConditionHandle *bc, struct tfParticleTypeHandle *t) {
    TFC_BIND_CHECKHANDLE(p);
    TFC_BIND_CHECKHANDLE(bc);
    TFC_BIND_CHECKHANDLE(t);
    return bind::boundaryCondition((Potential*)p->tfObj, (BoundaryCondition*)bc->tfObj, (ParticleType*)t->tfObj);
}

HRESULT tfBindForce(struct tfForceHandle *force, struct tfParticleTypeHandle *a_type) {
    TFC_BIND_CHECKHANDLE(force);
    TFC_BIND_CHECKHANDLE(a_type);
    return bind::force((Force*)force->tfObj, (ParticleType*)a_type->tfObj);
}

HRESULT tfBindForceS(struct tfForceHandle *force, struct tfParticleTypeHandle *a_type, const char *coupling_symbol) {
    TFC_BIND_CHECKHANDLE(force);
    TFC_BIND_CHECKHANDLE(a_type);
    TFC_PTRCHECK(coupling_symbol);
    return bind::force((Force*)force->tfObj, (ParticleType*)a_type->tfObj, coupling_symbol);
}

HRESULT tfBindBonds(
    struct tfPotentialHandle *potential,
    struct tfParticleListHandle *particles, 
    tfFloatP_t cutoff, 
    struct tfParticleTypeHandle **ppairsA, 
    struct tfParticleTypeHandle **ppairsB, 
    unsigned int numTypes, 
    tfFloatP_t *half_life, 
    tfFloatP_t *bond_energy, 
    struct tfBondHandleHandle **out, 
    unsigned int *numOut) 
{
    TFC_BIND_CHECKHANDLE(potential);
    TFC_BIND_CHECKHANDLE(particles);

    Potential *_potential = (Potential*)potential->tfObj;
    ParticleList *_particles = (ParticleList*)particles->tfObj;
    std::vector<std::pair<ParticleType*, ParticleType*>* > *pairs = NULL;
    if(ppairsA && ppairsB) {
        std::vector<std::pair<ParticleType*, ParticleType*>* > _pairs;
        struct tfParticleTypeHandle *pta, *ptb;
        for(unsigned int i = 0; i < numTypes; i++) {
            pta = ppairsA[i];
            ptb = ppairsB[i];
            TFC_BIND_CHECKHANDLE(pta);
            TFC_BIND_CHECKHANDLE(ptb);
            _pairs.push_back(new std::pair<ParticleType*, ParticleType*>(std::make_pair((ParticleType*)pta->tfObj, (ParticleType*)ptb->tfObj)));
        }
        pairs = &_pairs;
    }

    tfFloatP_t _half_life = half_life ? *half_life : std::numeric_limits<tfFloatP_t>::max();
    tfFloatP_t _bond_energy = bond_energy ? *bond_energy : std::numeric_limits<tfFloatP_t>::max();

    HRESULT result;
    if(out && numOut) {
        std::vector<BondHandle> _outv;
        result = bind::bonds(_potential, *_particles, cutoff, pairs, _half_life, _bond_energy, 0, &_outv);
        if(result == S_OK) {
            *numOut = _outv.size();
            tfBondHandleHandle *_out = (tfBondHandleHandle*)malloc(*numOut * sizeof(tfBondHandleHandle));
            for(unsigned int i = 0; i < _outv.size(); i++) {
                _out[i].tfObj = (void*)(new BondHandle(_outv[i]));
            }
            *out = _out;
        }
    }
    else {
        result = bind::bonds(_potential, *_particles, cutoff, pairs, _half_life, _bond_energy);
    }

    if(pairs) {
        for(unsigned int i = 0; i < pairs->size(); i++) 
            delete (*pairs)[i];
    }

    return result;
}
