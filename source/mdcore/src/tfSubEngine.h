/*******************************************************************************
 * This file is part of mdcore.
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

#ifndef _MDCORE_SOURCE_TFSUBENGINE_H_
#define _MDCORE_SOURCE_TFSUBENGINE_H_

#include <tf_port.h>


namespace TissueForge { 


    /**
     * @brief A SubEngine is a singleton object that injects dynamics into the Tissue Forge engine. 
     * It does not necessarily integrate any object in time, but can also 
     * simply add to the dynamics of existing Tissue Forge objects. 
     * Tissue Forge supports an arbitrary number of subengines with multi-threading and GPU support. 
     * 
     */
    struct SubEngine {

        /** Unique name of the engine. No two registered engines can have the same name. */
        const char* name;

        /**
         * @brief Register with the Tissue Forge engine.
         * 
         * @return HRESULT 
         */
        HRESULT registerEngine();

        /**
         * @brief First call before forces are calculated for a step. 
         * 
         * @return HRESULT 
         */
        virtual HRESULT preStepStart() { return S_OK; };

        /**
         * @brief Last call before forces are calculated for a step.
         * 
         * @return HRESULT 
         */
        virtual HRESULT preStepJoin() { return S_OK; };

        /**
         * @brief First call after forces are calculated, and before events are executed, for a step.
         * 
         * @return HRESULT 
         */
        virtual HRESULT postStepStart() { return S_OK; };

        /**
         * @brief Last call after forces are calculated, and before events are executed, for a step.
         * 
         * @return HRESULT 
         */
        virtual HRESULT postStepJoin() { return S_OK; };

        /**
         * @brief Called during termination of a simulation, just before shutdown of Tissue Forge engine.
         * 
         * @return HRESULT 
         */
        virtual HRESULT finalize() { return S_OK; };

    };

};

#endif // _MDCORE_SOURCE_TFSUBENGINE_H_