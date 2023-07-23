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

#include "tfStateVector.h"
#include "tfSpeciesList.h"
#include "tfSpeciesReactions.h"
#include <tfLogger.h>
#include <tfError.h>
#include <io/tfFIO.h>

#include <sbml/Species.h>

#include <iostream>
#include <sstream>


using namespace TissueForge;


// reset the species values based on the values specified in the species.
void state::StateVector::reset() {
    for(int i = 0; i < species->size(); ++i) {
        state::Species *s = species->item(i);
        FloatP_t value = 0.f;
        if(s->isSetInitialConcentration()) {
            value = s->getInitialConcentration();
        }
        fvec[i] = value;
    }
    this->reactions->reset();
}

static void statevector_copy_values(state::StateVector *newVec, const state::StateVector* oldVec) {
    for(int i = 0; i < oldVec->species->size(); ++i) {
        state::Species *species = oldVec->species->item(i);
        
        int j = newVec->species->index_of(species->getId().c_str());
        if(j >= 0) {
            newVec->fvec[j] = oldVec->fvec[i];
        }
    }
}

const std::string state::StateVector::str() const {
    std::stringstream  ss;
    
    ss << "StateVector([";
    for(int i = 0; i < size; ++i) {
        Species *s = species->item(i);
        ss << s->species->getId();
        ss << ":";
        ss << fvec[i];
        if(i+1 < size) {
            ss << ", ";
        }
    }
    ss << "])";
    return ss.str();
}

FloatP_t *state::StateVector::item(const int &i) {
    
    if(i >= 0 && i < size) return &fvec[i];
    else {
        tf_exp(std::runtime_error("state vector index out of range"));
        return NULL;
    }
}

void state::StateVector::setItem(const int &i, const FloatP_t &val) {
    if(i >= 0 && i < size) fvec[i] = val;
    else {
        tf_exp(std::runtime_error("state vector index out of range"));
    }
}

state::StateVector::StateVector() : 
    species(new state::SpeciesList()), 
    reactions(new state::SpeciesReactions(this)) 
{}

state::StateVector::StateVector(SpeciesList *_species, 
                                void *_owner, 
                                state::StateVector *existingStateVector, 
                                uint32_t flags, 
                                void *_data) 
{
    TF_Log(LOG_DEBUG) << "Creating state vector";

    this->species = _species;
    this->reactions = new SpeciesReactions(this);
    if(_owner) this->owner = _owner;
    
    this->size = _species->size();
    
    const int fvec_offset = 0;
    const int fvec_size = this->size * sizeof(FloatP_t);
    const int q_offset = fvec_offset + fvec_size;
    const int q_size = this->size * sizeof(FloatP_t);
    const int flags_offset = q_offset + q_size;
    const int flags_size = this->size * sizeof(int32_t);
    
    if(!_data) {
        this->flags |= STATEVECTOR_OWNMEMORY;
        this->data = malloc(fvec_size + q_size + flags_size);
        bzero(this->data, fvec_size + q_size + flags_size);
        this->fvec =          (FloatP_t*)((uint8_t*)this->data + fvec_offset);
        this->q =             (FloatP_t*)((uint8_t*)this->data + q_offset);
        this->species_flags = (uint32_t*)((uint8_t*)this->data + flags_offset);
    }

    // Copy from other state if provided; otherwise initialize from any available initial conditions
    if(existingStateVector) statevector_copy_values(this, existingStateVector);
    else {
        for(int i = 0; i < _species->size(); ++i) {
            auto _s = _species->item(i);
            if(_s->isSetInitialConcentration()) 
                this->fvec[i] = (FloatP_t)_s->getInitialConcentration();
        }
    }
    if(this->species->numReactions() > 0) {
        this->reactions = new SpeciesReactions(this);
        for(int i = 0; i < this->species->numReactions(); i++) {
            std::string modelName;
            SpeciesReactionDef rDef;
            this->species->reaction(i, modelName, rDef);

            TF_Log(LOG_TRACE) << "Creating reaction: " << modelName << ", " << rDef.stepSize << ", " << rDef.numSteps;

            this->reactions->create(modelName, rDef, false);
        }
    }
    
    for(int i = 0; i < _species->size(); ++i) {
        this->species_flags[i] = _species->item(i)->flags();
    }
}

state::StateVector::StateVector(const state::StateVector &other) : 
    state::StateVector(other.species, other.owner, const_cast<state::StateVector*>(&other), other.flags, 0)
{}

state::StateVector::~StateVector() {
    if(!owner) {
        delete species;
        species = 0;
        delete reactions;
        reactions = 0;
    }

    if(flags & STATEVECTOR_OWNMEMORY) {
        free(fvec);
    }
}

std::string state::StateVector::toString() {
    return io::toString(*this);
}

state::StateVector *state::StateVector::fromString(const std::string &str) {
    return io::fromString<state::StateVector*>(str);
}


namespace TissueForge::io { 

    template <>
    HRESULT toFile(const state::StateVector &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOTOEASY(fileElement, metaData, "flags", dataElement.flags);
        TF_IOTOEASY(fileElement, metaData, "size", dataElement.size);
        if(dataElement.species != NULL) {
            TF_IOTOEASY(fileElement, metaData, "species", *dataElement.species);
        }

        if(dataElement.size > 0) {
            std::vector<FloatP_t> fvec, q;
            std::vector<uint32_t> species_flags;
            for(unsigned int i = 0; i < dataElement.size; i++) {
                fvec.push_back(dataElement.fvec[i]);
                q.push_back(dataElement.q[i]);
                species_flags.push_back(dataElement.species_flags[i]);
            }
            
            TF_IOTOEASY(fileElement, metaData, "quantities", fvec);
            TF_IOTOEASY(fileElement, metaData, "fluxes", q);
            TF_IOTOEASY(fileElement, metaData, "species_flags", species_flags);
        }

        if(dataElement.reactions->models.size() > 0) {
            TF_IOTOEASY(fileElement, metaData, "reactions", *dataElement.reactions);
        }

        fileElement.get()->type = "StateVector";

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, state::StateVector **dataElement) {
        uint32_t flags;
        TF_IOFROMEASY(fileElement, metaData, "flags", &flags);
        
        uint32_t size;
        TF_IOFROMEASY(fileElement, metaData, "size", &size);

        state::SpeciesList *species = new state::SpeciesList();
        std::vector<FloatP_t> fvec, q;
        std::vector<uint32_t> species_flags;

        if(size > 0) {
            TF_IOFROMEASY(fileElement, metaData, "species", species);
            TF_IOFROMEASY(fileElement, metaData, "quantities", &fvec);
            TF_IOFROMEASY(fileElement, metaData, "fluxes", &q);
            TF_IOFROMEASY(fileElement, metaData, "species_flags", &species_flags);
        }

        *dataElement = new state::StateVector(species, 0, 0, flags);

        if(size > 0) {
            for(unsigned int i = 0; i < size; i++) {

                (*dataElement)->fvec[i] = fvec[i];
                (*dataElement)->q[i] = q[i];
                (*dataElement)->species_flags[i] = species_flags[i];

            }
        }

        IOChildMap fec = fileElement.children(fileElement);
        if(fec.find("reactions") != fec.end()) {
            TF_IOFROMEASY(fileElement, metaData, "reactions", (*dataElement)->reactions);
        }

        return S_OK;
    }

};
