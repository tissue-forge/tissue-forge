/*******************************************************************************
 * This file is part of mdcore.
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

#include <tfSecreteUptake.h>
#include <tfParticle.h>
#include <tfError.h>
#include <state/tfSpeciesList.h>
#include <state/tfStateVector.h>
#include <tfParticleList.h>
#include <tf_metrics.h>
#include <tfEngine.h>
#include <iostream>


using namespace TissueForge;



HRESULT TissueForge::Secrete_AmountToParticles(struct state::SpeciesValue *species,
        FPTYPE amount, uint16_t nr_parts,
        int32_t *parts, FPTYPE *secreted)
{
    state::StateVector *stateVector = species->state_vector;
    state::Species *s = stateVector->species->item(species->index);
    const std::string& speciesName = s->getId();
    
    FPTYPE amountToRemove = amount < stateVector->fvec[species->index] ? amount : stateVector->fvec[species->index];
    
    struct ParticleId {
        Particle *part;
        int32_t index;
    };
    
    std::vector<ParticleId> pvec;
    
    for(int i = 0; i < nr_parts; ++i) {
        Particle *p = _Engine.s.partlist[parts[i]];
        
        int index;
        
        if(p && p->state_vector && (index = p->state_vector->species->index_of(speciesName.c_str())) >= 0) {
            pvec.push_back({p, index});
        }
    }
    
    if(pvec.size() > 0) {
        FPTYPE amountPer = amountToRemove / pvec.size();
        for(ParticleId& p : pvec) {
            p.part->state_vector->fvec[p.index] += amountPer;
        }
        stateVector->fvec[species->index] -= amountToRemove;
        if(secreted) {
            *secreted = amountToRemove;
        }
    }
    
    return S_OK;
}

HRESULT TissueForge::Secrete_AmountWithinDistance(struct state::SpeciesValue *species,
        FPTYPE amount, FPTYPE radius,
        const std::set<short int> *typeIds, FPTYPE *secreted)
{
    Particle *part = (Particle*)species->state_vector->owner;
    uint16_t nr_parts = 0;
    int32_t *parts = NULL;
    
    metrics::particleNeighbors(part, radius, typeIds, &nr_parts, &parts);
    
    return Secrete_AmountToParticles(species, amount, nr_parts, parts, secreted);
}


FPTYPE TissueForge::SecreteUptake::secrete(state::SpeciesValue *species, const FPTYPE &amount, const ParticleList &to) {
    FPTYPE secreted = 0;
    try{
        if(FAILED(Secrete_AmountToParticles(species, amount, to.nr_parts, to.parts, &secreted))) return FPTYPE_ZERO;
    }
    catch(const std::exception &e) {
        
    }
    return secreted;
}

FPTYPE TissueForge::SecreteUptake::secrete(state::SpeciesValue *species, const FPTYPE &amount, const FPTYPE &distance) {
    FPTYPE secreted = 0;
    
    Particle *part = (Particle*)species->state_vector->owner;
    if(!part) {
        tf_exp(std::runtime_error("species state vector has no owner"));
        return 0.0;
    }
    
    try{
        // take into account the radius of this particle.
        FPTYPE radius = (FPTYPE)part->radius + distance;
        std::set<short int> ids = (std::set<short int>)ParticleType::particleTypeIds();
        if(FAILED(Secrete_AmountWithinDistance(species, amount, radius, &ids, &secreted))) return FPTYPE_ZERO;
    }
    catch(const std::exception &e) {
        
    }
    
    return secreted;
}
