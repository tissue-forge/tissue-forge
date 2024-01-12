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

/**
 * @file tfStateVector.h
 * 
 */

#ifndef _SOURCE_STATE_TFSTATEVECTOR_H_
#define _SOURCE_STATE_TFSTATEVECTOR_H_

#include <tf_port.h>
#include <io/tf_io.h>
#include <stdio.h>
#include <string>

#include "tfSpeciesList.h"


namespace TissueForge {


    namespace state {


        enum StateVectorFlags {
            STATEVECTOR_NONE            = 0,
            STATEVECTOR_OWNMEMORY       = 1 << 0,
        };

        /**
         * @brief A state vector of an object. 
         */
        struct CAPI_EXPORT StateVector {
            uint32_t flags;
            uint32_t size;

            /**
             * @brief Species of the state vector
             */
            struct TissueForge::state::SpeciesList *species;
            
            /**
             * owner of this state vector, usually a
             * Particle, but we leave the door open for other
             * kinds of things.
             */
            void *owner;
            
            void* data;
            
            // vector of values
            FloatP_t *fvec;
            
            // vector of fluxes
            FloatP_t *q;
            
            // vector of species flags
            uint32_t *species_flags;
            
            // reset the species values based on the values specified in the species.
            void reset();
            const std::string str() const;

            FloatP_t *item(const int &i);
            void setItem(const int &i, const FloatP_t &val);

            StateVector();
            StateVector(TissueForge::state::SpeciesList *species, 
                        void *owner=NULL, 
                        StateVector *existingStateVector=NULL,
                        uint32_t flags=STATEVECTOR_NONE, 
                        void *data=NULL);
            StateVector(const StateVector &other);
            ~StateVector();

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
             * @return StateVector* 
             */
            static StateVector *fromString(const std::string &str);
        };

    }

    namespace io { 

        template <>
        HRESULT toFile(const TissueForge::state::StateVector &dataElement, const MetaData &metaData, IOElement &fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::state::StateVector **dataElement);

    }

};

#endif // _SOURCE_STATE_TFSTATEVECTOR_H_