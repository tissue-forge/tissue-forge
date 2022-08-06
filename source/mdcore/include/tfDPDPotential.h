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

/**
 * @file tfDPDPotential.h
 * 
 */

#pragma once

#include "tfPotential.h"

#ifndef _MDCORE_INCLUDE_TFDPDPOTENTIAL_H_
#define _MDCORE_INCLUDE_TFDPDPOTENTIAL_H_


namespace TissueForge { 


    struct CAPI_EXPORT DPDPotential : public Potential {
        
        /** strength of conserative interaction */
        FPTYPE alpha;
        
        /** strength of dissapative interaction */
        FPTYPE gamma;
        
        /** strength of random interaction */
        FPTYPE sigma;

        DPDPotential(FPTYPE alpha, FPTYPE gamma, FPTYPE sigma, FPTYPE cutoff, bool shifted);

        /**
         * @brief Convert basic potential to DPD. 
         * 
         * If the basic potential is not DPD, then NULL is returned. 
         * 
         * @param pot 
         * @return DPDPotential* 
         */
        static DPDPotential *fromPot(Potential *pot);

        /**
         * @brief Get a JSON string representation
         * 
         * @return std::string 
         */
        std::string toString();

        /**
         * @brief Create from a JSON string representation
         * 
         * @param str 
         * @return Potential* 
         */
        static DPDPotential *fromString(const std::string &str);
    };


    CPPAPI_FUNC(DPDPotential*) DPDPotential_fromStr(const std::string &str);

};

#endif // _MDCORE_INCLUDE_TFDPDPOTENTIAL_H_