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

#ifndef _SOURCE_STATE_TFSPECIESVALUE_H_
#define _SOURCE_STATE_TFSPECIESVALUE_H_

#include <tf_port.h>
#include "tfSpecies.h"


namespace TissueForge {


    struct ParticleList;


    namespace state {


        struct StateVector;


        /**
         * @brief A working valued-object of an underlying Species attached to an object. 
         */
        struct CAPI_EXPORT SpeciesValue
        {
            struct TissueForge::state::StateVector *state_vector;
            uint32_t index;

            TissueForge::state::Species *species();

            FloatP_t getValue() const;
            void setValue(const FloatP_t &_value);
            bool getBoundaryCondition();
            int setBoundaryCondition(const int &value);
            FloatP_t getInitialAmount();
            int setInitialAmount(const FloatP_t &value);
            FloatP_t getInitialConcentration();
            int setInitialConcentration(const FloatP_t &value);
            bool getConstant();
            int setConstant(const int &value);

            /**
             * @brief Secrete this species into a neighborhood. 
             * 
             * @param amount Amount to secrete. 
             * @param to List of particles to secrete to. 
             * @return FloatP_t Amount actually secreted, accounting for availability and other subtleties. 
             */
            FloatP_t secrete(const FloatP_t &amount, const TissueForge::ParticleList &to);

            /**
             * @brief Secrete this species into a neighborhood. 
             * 
             * @param amount Amount to secrete. 
             * @param distance Neighborhood distance. 
             * @return FloatP_t Amount actually secreted, accounting for availability and other subtleties. 
             */
            FloatP_t secrete(const FloatP_t &amount, const FloatP_t &distance);

            SpeciesValue(struct TissueForge::state::StateVector *state_vector, uint32_t index);
        };

}}

#endif // _SOURCE_STATE_TFSPECIESVALUE_H_