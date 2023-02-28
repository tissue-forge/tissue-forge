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

#include "tfSpecies.h"
#include "tfSpeciesList.h"

#include <tfLogger.h>
#include <io/tfFIO.h>

#include <sstream>
#include <iostream>
#include <iterator>


using namespace TissueForge;


int32_t state::SpeciesList::index_of(const std::string &s) 
{
    int32_t result = -1;
    
    auto i = species_map.find(s);
    
    if(i != species_map.end()) {
        result = std::distance(species_map.begin(), i);
    }
    
    return result;
}

int32_t state::SpeciesList::size() 
{
    return species_map.size();
}

state::Species* state::SpeciesList::item(const std::string &s) 
{
    auto i = species_map.find(s);
    if(i != species_map.end()) {
        return i->second;
    }
    return NULL;
}

state::Species* state::SpeciesList::item(int32_t index) 
{
    if(index < species_map.size()) {
        auto i = species_map.begin();
        i = std::next(i, index);
        return i->second;
    }
    return NULL;
}

HRESULT state::SpeciesList::insert(state::Species* s)
{
    TF_Log(LOG_DEBUG) << "Inserting species: " << s->getId();
    
    species_map.emplace(s->getId(), s);

    TF_Log(LOG_DEBUG) << size();
    TF_Log(LOG_DEBUG) << str();
    return S_OK;
}

HRESULT state::SpeciesList::insert(const std::string &s) {
    return insert(new Species(s));
}

std::string state::SpeciesList::str() {
    std::stringstream  ss;
    
    ss << "SpeciesList([";
    for(int i = 0; i < size(); ++i) {
        Species *s = item(i);
        ss << "'" << s->getId() << "'";

        if(i+1 < size()) {
            ss << ", ";
        }
    }
    ss << "])";
    return ss.str();
}

state::SpeciesList::~SpeciesList() {
    
    for (auto &i : species_map) {
        delete i.second;
    }
    species_map.clear();

}

std::string state::SpeciesList::toString() {
    return io::toString(*this);
}

state::SpeciesList *state::SpeciesList::fromString(const std::string &str) {
    return new state::SpeciesList(io::fromString<state::SpeciesList>(str));
}


namespace TissueForge::io { 


    template <>
    HRESULT toFile(const state::SpeciesList &dataElement, const MetaData &metaData, IOElement &fileElement) { 
        state::SpeciesList *sList = const_cast<state::SpeciesList*>(&dataElement);
        
        auto numSpecies = sList->size();
        std::vector<state::Species> species;

        for(unsigned int i = 0; i < numSpecies; i++) 
            species.push_back(*sList->item(i));

        TF_IOTOEASY(fileElement, metaData, "species", species);

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, state::SpeciesList *dataElement) {
        std::vector<state::Species> species;
        TF_IOFROMEASY(fileElement, metaData, "species", &species);

        for(auto s : species) 
            dataElement->insert(new state::Species(s));

        return S_OK;
    }

};
