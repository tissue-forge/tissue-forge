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
 * @file tfSpeciesReactionDef.h
 * 
 */

#ifndef _SOURCE_STATE_TFSPECIESREACTIONDEF_H_
#define _SOURCE_STATE_TFSPECIESREACTIONDEF_H_

#include <tf_port.h>
#include <io/tf_io.h>

#include <string>
#include <tuple>
#include <vector>


#define TF_ROADRUNNER_DEFAULT_STEPSIZE 1.0
#define TF_ROADRUNNER_DEFAULT_NUMSTEPS 10


namespace TissueForge {


    namespace state {


        /**
         * @brief Enums for all supported integrators
         */
        enum SpeciesReactionIntegrators {
            SPECIESREACTIONINT_NONE         = 0,
            SPECIESREACTIONINT_CVODE,
            SPECIESREACTIONINT_EULER,
            SPECIESREACTIONINT_GILLESPIE,
            SPECIESREACTIONINT_NLEQ1,
            SPECIESREACTIONINT_NLEQ2,
            SPECIESREACTIONINT_RK4,
            SPECIESREACTIONINT_RK45
        };


        /**
         * @brief Reaction kinetics model static definition. 
         * 
         * @details Essentially a lightweight container for model definitions 
         * that can be used to efficiently create \ref SpeciesReaction instances. 
         * 
         * Tissue Forge maintains synchronization between \ref StateVector and 
         * \ref SpeciesReaction values through mapping by name. For each \ref Species, 
         * a corresponding reaction kinetics model variable can be synchronized by 
         * model variable name. 
         * 
         * Includes convenience features to use models specified in Antimony. 
         * Models are translated to SBML using libAntimony. 
         */
        struct CAPI_EXPORT SpeciesReactionDef {

            /**
             * @brief SBML model string or file path
             */
            std::string uriOrSBML;

            /**
             * @brief List of mapped species names
             */
            std::vector<std::string> speciesNames;

            /**
             * @brief List of mapped model variable names
             */
            std::vector<std::string> modelNames;

            /**
             * @brief Period of a simulation step according to the model
             */
            double stepSize = TF_ROADRUNNER_DEFAULT_STEPSIZE;

            /**
             * @brief Number of substeps per simulation step
             */
            int numSteps = TF_ROADRUNNER_DEFAULT_NUMSTEPS;

            /**
             * @brief Simulation integrator enum
             */
            SpeciesReactionIntegrators integrator = SPECIESREACTIONINT_CVODE;

            SpeciesReactionDef() {};

            /**
             * @param uriOrSBML SBML string or file path
             * @param nameMap map of species names to model variable names
             */
            SpeciesReactionDef(
                std::string uriOrSBML, 
                const std::vector<std::pair<std::string, std::string > >& nameMap
            );

            /**
             * @brief Construct an instance using an Antimony model string
             * 
             * @param modelString Antimony model string
             * @param nameMap map of species names to model variable names
             */
            static SpeciesReactionDef fromAntimonyString(
                const std::string& modelString, 
                const std::vector<std::pair<std::string, std::string > >& nameMap
            );

            /**
             * @brief Construct an instance using an Antimony model file
             * 
             * @param modelString path to a file containing an Antimony model
             * @param nameMap map of species names to model variable names
             */
            static SpeciesReactionDef fromAntimonyFile(
                const std::string& modelFilePath, 
                const std::vector<std::pair<std::string, std::string > >& nameMap
            );

            /**
             * @brief Get the name of an integrator by enum
             */
            static std::string integratorName(const SpeciesReactionIntegrators& sri);

            /**
             * @brief Get the enum of an integrator by name
             */
            static SpeciesReactionIntegrators integratorEnum(const std::string& name);

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
             * @return SpeciesReactionDef* 
             */
            static SpeciesReactionDef fromString(const std::string &str);
        };

    }

    namespace io {

        template <>
        HRESULT toFile(const TissueForge::state::SpeciesReactionDef &dataElement, const MetaData &metaData, IOElement &fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::state::SpeciesReactionDef *dataElement);

    }

}

#endif // _SOURCE_STATE_TFSPECIESREACTIONDEF_H_