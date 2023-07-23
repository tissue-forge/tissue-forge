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
 * @file tfSpeciesReaction.h
 * 
 */

#ifndef _SOURCE_STATE_TFSPECIESREACTION_H_
#define _SOURCE_STATE_TFSPECIESREACTION_H_

#include "tfSpeciesReactionDef.h"

#include <tf_port.h>
#include <io/tf_io.h>

#include <string>
#include <tuple>
#include <vector>


namespace rr {
    class RoadRunner;
}


namespace TissueForge {


    namespace state {


        struct StateVector;


        /**
         * @brief Tissue Forge reaction kinetics model executed in libRoadRunner. 
         * 
         * @details Each instance has a \ref StateVector owner, 
         * and the instance can maintain synchronization 
         * betwen \ref StateVector values and model variable values by name mapping. 
         * Synchronization is maintained through flux transport such that reaction diffusion 
         * can be implemented through flux transport + reaction kinetics modeling. 
         * 
         * Models are specified in SBML. For using models specified in Antimony, 
         * see \ref SpeciesReactionDef. 
         */
        struct CAPI_EXPORT SpeciesReaction {

            /**
             * @brief Owner
             */
            StateVector* owner;

            SpeciesReaction() {}
            
            /**
             * @param svec owner
             * @param uriOrSBML SBML string or file path
             * @param nameMap map of species names to model variable names
             * @param mapFrom flag to copy values from model to species. false performs the opposite. 
             */
            SpeciesReaction(
                StateVector* svec, 
                const std::string& uriOrSBML, 
                const std::vector<std::pair<std::string, std::string > >& nameMap, 
                const bool& mapFrom=false
            );

            /**
             * @param svec owner
             * @param uriOrSBML SBML string or file path
             * @param mapFrom flag to copy values from model to species. false performs the opposite. 
             */
            SpeciesReaction(StateVector* svec, const std::string& uriOrSBML, const bool& mapFrom=false);

            /**
             * @param svec owner
             * @param rDef model definition
             * @param mapFrom flag to copy values from model to species. false performs the opposite. 
             */
            SpeciesReaction(StateVector* svec, const SpeciesReactionDef& rDef, const bool& mapFrom=false);

            /**
             * @param svec owner
             * @param other another instance
             * @param mapFrom flag to copy values from model to species. false performs the opposite. 
             */
            SpeciesReaction(StateVector* svec, const SpeciesReaction& other, const bool& mapFrom=false);

            ~SpeciesReaction();

            /**
             * @brief Integrate over a period
             * 
             * @param until period over which to integrate the model. Default is the universe time step
             */
            HRESULT step(const double& until=-1.0);

            /**
             * @brief Integrate over a number of steps
             * 
             * @param numSteps Number of steps over which to integrate
             */
            HRESULT step(const unsigned int& numSteps=1);

            /**
             * @brief Integrate over a period of time according to the universe
             * 
             * @param univdt Period of universe time over which to integrate
             */
            HRESULT stepT(const double& univdt);

            /**
             * @brief Get cached fluxes. For internal use only.  
             */
            std::vector<FloatP_t> getCachedFluxes();

            /**
             * @brief Reset cached fluxes. For internal use only.
             */
            HRESULT resetCachedFluxes();

            /**
             * @brief Reset the model to its initial state.
             */
            HRESULT reset();

            /**
             * @brief Map a species name to a model variable name
             * 
             * @param speciesName Species name
             * @param modelName Model variable name
             */
            HRESULT mapName(const std::string& speciesName, const std::string& modelName);

            /**
             * @brief Map a species to a model variable of the same name
             * 
             * @param speciesName Species name
             */
            HRESULT mapName(const std::string& speciesName);

            /**
             * @brief Get list of all mapped species names
             */
            std::vector<std::string> getSpeciesNames();

            /**
             * @brief Get list of all mapped model variable names
             */
            std::vector<std::string> getModelNames();

            /**
             * @brief Reset name mapping
             */
            HRESULT resetMap();

            /**
             * @brief Map values from species to model variables
             */
            HRESULT mapValuesTo();

            /**
             * @brief Map values from species to model variables
             */
            HRESULT mapValuesFrom();

            /**
             * @brief Set the period of a step according to the model
             */
            HRESULT setStepSize(const double& stepSize);

            /**
             * @brief Get the period of a step according to the model
             */
            double getStepSize();

            /**
             * @brief Set the number of substeps per simulation step
             */
            HRESULT setNumSteps(const int& numSteps);

            /**
             * @brief Get the number of substeps per simulation step
             */
            int getNumSteps();

            /**
             * @brief Set the current time according to the model
             */
            HRESULT setCurrentTime(const double& currentTime);

            /**
             * @brief Get the current time according to the model
             */
            double getCurrentTime();

            /**
             * @brief Set the value of a model variable by name
             * 
             * @param name Model variable name
             * @param value Model variable value
             */
            HRESULT setModelValue(const std::string& name, const double& value);

            /**
             * @brief Get the value of a model variable by name
             * 
             * @param name Model variable name
             */
            double getModelValue(const std::string& name);

            /**
             * @brief Test whether the model has a variable by name
             * 
             * @param name Model variable name
             */
            bool hasModelValue(const std::string& name);

            /**
             * @brief Get the name of the current integrator
             */
            std::string getIntegratorName();

            /**
             * @brief Set the current integrator by name
             * 
             * @param name Integrator name
             */
            HRESULT setIntegratorName(const std::string& name);

            /**
             * @brief Test whether using an integrator by enum
             * 
             * @param sri Integrator enum to test
             */
            bool hasIntegrator(const SpeciesReactionIntegrators& sri);

            /**
             * @brief Get the current integrator enum
             */
            SpeciesReactionIntegrators getIntegrator();

            /**
             * @brief Set the current integrator by enum
             * 
             * @param sri Integrator enum
             */
            HRESULT setIntegrator(const SpeciesReactionIntegrators& sri);

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
             * @return SpeciesReaction* 
             */
            static SpeciesReaction fromString(const std::string &str);

            friend HRESULT io::toFile(const SpeciesReaction &dataElement, const io::MetaData &metaData, io::IOElement &fileElement);
            friend HRESULT io::toFile(SpeciesReaction *dataElement, const io::MetaData &metaData, io::IOElement &fileElement);
            friend HRESULT io::fromFile(const io::IOElement &fileElement, const io::MetaData &metaData, SpeciesReaction *dataElement);
            friend HRESULT io::fromFile(const io::IOElement &fileElement, const io::MetaData &metaData, SpeciesReaction **dataElement);

        private:

            rr::RoadRunner* rr;
            std::vector<int> svecIndices;
            std::vector<std::string> modelNames;
            std::vector<FloatP_t> cachedFluxes;

            double currentTime = 0.0;
            double dt = TF_ROADRUNNER_DEFAULT_STEPSIZE;
            int numSteps = TF_ROADRUNNER_DEFAULT_NUMSTEPS;
        };

    }

    namespace io {

        template <>
        HRESULT toFile(const TissueForge::state::SpeciesReaction &dataElement, const MetaData &metaData, IOElement &fileElement);

        template <>
        HRESULT toFile(TissueForge::state::SpeciesReaction *dataElement, const MetaData &metaData, IOElement &fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::state::SpeciesReaction *dataElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::state::SpeciesReaction **dataElement);

    }
    
}

#endif // _SOURCE_STATE_TFSPECIESREACTION_H_