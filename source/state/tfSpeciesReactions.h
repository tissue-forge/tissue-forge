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
 * @file tfSpeciesReactions.h
 * 
 */

#ifndef _SOURCE_STATE_TFSPECIESREACTIONS_H_
#define _SOURCE_STATE_TFSPECIESREACTIONS_H_

#include "tfSpeciesReactionDef.h"
#include "tfSpeciesReaction.h"

#include <unordered_map>


namespace TissueForge {


    namespace state {


        struct StateVector;


        /**
         * @brief A container holding the reactions of a @ref Species
        */
        struct CAPI_EXPORT SpeciesReactions {

            /**
             * @brief Map of reaction models by name
             */
            std::unordered_map<std::string, SpeciesReaction*> models;

            /**
             * @brief Owner
             */
            StateVector* owner;

            SpeciesReactions() {};

            /**
             * @param svec owner
             */
            SpeciesReactions(StateVector* svec);
            ~SpeciesReactions();

            /**
             * @brief Integrate all models a number of steps
             * 
             * @param numSteps Number of steps over which to integrate
             */
            HRESULT step(const unsigned int& numSteps=1);

            /**
             * @brief Integrate all models over a period
             * 
             * @param univdt Period of time according to the universe over which to integrate
             */
            HRESULT stepT(const double& univdt);

            /**
             * @brief Get cached fluxes. For internal use only. 
             */
            std::vector<FloatP_t> getCachedFluxes();

            /**
             * @brief Reset all models to the initial state
             */
            HRESULT reset();

            /**
             * @brief Create a model
             * 
             * @param modelName Model name
             * @param rDef Model definition
             * @param mapFrom Flag to map values from the model to species when true. false performs the opposite. 
             */
            HRESULT create(const std::string& modelName, const SpeciesReactionDef& rDef, const bool& mapFrom=false);

            /**
             * @brief Create a model
             * 
             * @param modelName Model name
             * @param model Model to copy
             * @param mapFrom Flag to map values from the model to species when true. false performs the opposite. 
             */
            HRESULT create(const std::string& modelName, const SpeciesReaction& model, const bool& mapFrom=false);

            /**
             * @brief Copy from an instance
             * 
             * @param other Another instance
             */
            HRESULT copyFrom(const SpeciesReactions& other);

            /**
             * @brief Map values from species to model variables
             */
            HRESULT mapValuesTo();

            /**
             * @brief Map values from species to model variables
             */
            HRESULT mapValuesFrom();

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
             * @return SpeciesReactions* 
             */
            static SpeciesReactions fromString(const std::string &str);

            #ifdef SWIGPYTHON

            /**
             * @brief Get a model by name
             */
            SpeciesReaction* _getModel(const std::string& key) {
                auto mitr = models.find(key);
                if(mitr == models.end()) return NULL;
                return mitr->second;
            }

            /**
             * @brief List of model names
             */
            std::vector<std::string> _modelNames() {
                std::vector<std::string> result;
                for(auto& mitr : models) 
                    result.push_back(mitr.first);
                return result;
            }

            #endif
        };

    }

    namespace io {

        template <>
        HRESULT toFile(const TissueForge::state::SpeciesReactions &dataElement, const MetaData &metaData, IOElement &fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::state::SpeciesReactions *dataElement);

    }
}

#endif // _SOURCE_STATE_TFSPECIESREACTIONS_H_