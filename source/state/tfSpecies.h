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

/**
 * @file tfSpecies.h
 * 
 */

#ifndef _SOURCE_STATE_TFSPECIES_H_
#define _SOURCE_STATE_TFSPECIES_H_

#include <tf_port.h>
#include <io/tf_io.h>
#include <string>


namespace libsbml {
    class Species;
    class SBMLNamespaces;
};


namespace TissueForge {


    namespace state {


        enum SpeciesFlags {

            SPECIES_BOUNDARY  = 1 << 0,
            SPECIES_SUBSTANCE = 1 << 2,
            SPECIES_CONSTANT  = 1 << 3,
            SPECIES_KONSTANT  = SPECIES_BOUNDARY | SPECIES_CONSTANT
        };

        /**
         * @brief The Tissue Forge species object. 
         * 
         * Mostly, this is a wrap of libSBML Species. 
         */
        struct CAPI_EXPORT Species {
            libsbml::Species *species;
            int32_t flags() const;
            std::string str() const;
            void initDefaults();

            const std::string getId() const;
            int setId(const char *sid);
            const std::string getName() const;
            int setName(const char *name);
            const std::string getSpeciesType() const;
            int setSpeciesType(const char *sid);
            const std::string getCompartment() const;
            int setCompartment(const char *sid);
            FloatP_t getInitialAmount() const;
            int setInitialAmount(FloatP_t value);
            FloatP_t getInitialConcentration() const;
            int setInitialConcentration(FloatP_t value);
            const std::string getSubstanceUnits() const;
            int setSubstanceUnits(const char *sid);
            const std::string getSpatialSizeUnits() const;
            int setSpatialSizeUnits(const char *sid);
            const std::string getUnits() const;
            int setUnits(const char *sname);
            bool getHasOnlySubstanceUnits() const;
            int setHasOnlySubstanceUnits(int value);
            bool getBoundaryCondition() const;
            int setBoundaryCondition(int value);
            int getCharge() const;
            int setCharge(int value);
            bool getConstant() const;
            int setConstant(int value);
            const std::string getConversionFactor() const;
            int setConversionFactor(const char *sid);
            bool isSetId() const;
            bool isSetName() const;
            bool isSetSpeciesType() const;
            bool isSetCompartment() const;
            bool isSetInitialAmount() const;
            bool isSetInitialConcentration() const;
            bool isSetSubstanceUnits() const;
            bool isSetSpatialSizeUnits() const;
            bool isSetUnits() const;
            bool isSetCharge() const;
            bool isSetConversionFactor() const;
            bool isSetConstant() const;
            bool isSetBoundaryCondition() const;
            bool isSetHasOnlySubstanceUnits() const;
            int unsetId();
            int unsetName();
            int unsetConstant();
            int unsetSpeciesType();
            int unsetInitialAmount();
            int unsetInitialConcentration();
            int unsetSubstanceUnits();
            int unsetSpatialSizeUnits();
            int unsetUnits();
            int unsetCharge();
            int unsetConversionFactor();
            int unsetCompartment();
            int unsetBoundaryCondition();
            int unsetHasOnlySubstanceUnits();
            int hasRequiredAttributes();

            Species();
            Species(const std::string &s);
            Species(const Species &other);
            ~Species();

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
             * @return Species* 
             */
            static Species *fromString(const std::string &str);
        };


        CAPI_FUNC(libsbml::SBMLNamespaces*) getSBMLNamespaces();

    }

    namespace io { 

        template <>
        HRESULT toFile(const TissueForge::state::Species &dataElement, const MetaData &metaData, IOElement &fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, TissueForge::state::Species *dataElement);

    };

}

#endif // _SOURCE_STATE_TFSPECIES_H_
