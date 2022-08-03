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

#ifndef _MDCORE_INCLUDE_TFSECRETEUPTAKE_H_
#define _MDCORE_INCLUDE_TFSECRETEUPTAKE_H_

#include <mdcore_config.h>
#include <state/tfSpeciesValue.h>
#include "tfParticleList.h"

#include <set>


namespace TissueForge { 


    // Simple methods container
    struct CAPI_EXPORT SecreteUptake {

        static FPTYPE secrete(state::SpeciesValue *species, const FPTYPE &amount, const ParticleList &to);
        static FPTYPE secrete(state::SpeciesValue *species, const FPTYPE &amount, const FPTYPE &distance);

    };

    CAPI_FUNC(HRESULT) Secrete_AmountToParticles(
        struct state::SpeciesValue* species,
        FPTYPE amount,
        uint16_t nr_parts, int32_t *parts,
        FPTYPE *secreted
    );

    CAPI_FUNC(HRESULT) Secrete_AmountWithinDistance(
        struct state::SpeciesValue* species,
        FPTYPE amount,
        FPTYPE radius,
        const std::set<short int> *typeIds,
        FPTYPE *secreted
    );

};

#endif // _MDCORE_INCLUDE_TFSECRETEUPTAKE_H_