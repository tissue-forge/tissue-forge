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

#ifndef _SOURCE_STATE_TFSPECIESLIST_H_
#define _SOURCE_STATE_TFSPECIESLIST_H_

#include "tfSpecies.h"
#include <io/tf_io.h>
#include <tf_port.h>
#include <string>
#include <map>


namespace TissueForge {


    namespace state {


        struct CAPI_EXPORT SpeciesList
        {
            /**
             * @brief Get the index of a species name
             * 
             * @param s species name
             * @return int32_t >= 0 on sucess, -1 on failure
             */
            int32_t index_of(const std::string &s);
            
            /**
             * @brief Get the size of the species
             * 
             * @return int32_t 
             */
            int32_t size();
            
            /**
             * @brief Get a species by index
             * 
             * @param index index of the species
             * @return Species* 
             */
            TissueForge::state::Species *item(int32_t index);
            
            /**
             * @brief Get a species by name
             * 
             * @param s name of species
             * @return Species* 
             */
            TissueForge::state::Species *item(const std::string &s);
            
            /**
             * @brief Insert a species
             * 
             * @return HRESULT 
             */
            HRESULT insert(TissueForge::state::Species *s);

            /**
             * @brief Insert a species by name
             * 
             * @param s name of the species
             * @return HRESULT 
             */
            HRESULT insert(const std::string &s);

            /**
             * @brief Get a string representation
             * 
             * @return std::string 
             */
            std::string str();

            SpeciesList() {};

            ~SpeciesList();

            /**
             * @brief Get a JSON string representation
             * 
             * @return std::string 
             */
            std::string toString();

            /**
             * @brief Create from a JSON string representation. 
             * 
             * @param str 
             * @return SpeciesList* 
             */
            static SpeciesList *fromString(const std::string &str);
            
        private:
            
            typedef std::map<std::string, TissueForge::state::Species*> Map;
            Map species_map;
        };

    }

    namespace io { 

        template <>
        HRESULT toFile(const TissueForge::state::SpeciesList &dataElement, const MetaData &metaData, IOElement &fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::state::SpeciesList *dataElement);

    };

}


#endif // _SOURCE_STATE_TFSPECIESLIST_H_