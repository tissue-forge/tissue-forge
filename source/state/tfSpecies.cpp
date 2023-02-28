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

#include "tfSpecies.h"
#include <tfLogger.h>
#include <tfError.h>
#include <io/tfFIO.h>

#include <sbml/Species.h>
#include <sbml/SBMLNamespaces.h>

#include <iostream>
#include <regex>


static libsbml::SBMLNamespaces *sbmlns = NULL;


using namespace TissueForge;


libsbml::SBMLNamespaces* state::getSBMLNamespaces() {
    if(!sbmlns) {
        sbmlns = new libsbml::SBMLNamespaces();
    }
    return sbmlns;
}

const std::string state::Species::getId() const
{
    if (species) return this->species->getId();
    return "";
}

int state::Species::setId(const char *sid)
{
    if (species) return species->setId(std::string(sid));
    return -1;
}

const std::string state::Species::getName() const
{
    if(species && species->isSetName()) return species->getName();
    return "";
}

int state::Species::setName(const char *name)
{
    if (species) return species->setName(std::string(name));
    return -1;
}

const std::string state::Species::getSpeciesType() const
{
    if (species) return species->getSpeciesType();
    return "";
}

int state::Species::setSpeciesType(const char *sid)
{
    if(species) return species->setSpeciesType(sid);
    return -1;
}

const std::string state::Species::getCompartment() const
{
    if(species) return species->getCompartment();
    return "";
}

FloatP_t state::Species::getInitialAmount() const
{
    if(species && species->isSetInitialAmount()) return species->getInitialAmount();
    return 0.0;
}

int state::Species::setInitialAmount(FloatP_t value)
{
    if(species) return species->setInitialAmount(value);
    return -1;
}

FloatP_t state::Species::getInitialConcentration() const
{
    if(species && species->isSetInitialConcentration()) return species->getInitialConcentration();
    return 0.0;
}

int state::Species::setInitialConcentration(FloatP_t value)
{
    if(species) return species->setInitialConcentration(value);
    return -1;
}

const std::string state::Species::getSubstanceUnits() const
{
    if(species) return species->getSubstanceUnits();
    return "";
}

const std::string state::Species::getSpatialSizeUnits() const
{
    if(species) return species->getSpatialSizeUnits();
    return "";
}

const std::string state::Species::getUnits() const
{
    if(species) return species->getUnits();
    return "";
}

bool state::Species::getHasOnlySubstanceUnits() const
{
    if(species && species->isSetHasOnlySubstanceUnits()) return species->getHasOnlySubstanceUnits();
    return false;
}

int state::Species::setHasOnlySubstanceUnits(int value)
{
    if(species) return species->setHasOnlySubstanceUnits((bool)value);
    return -1;
}

bool state::Species::getBoundaryCondition() const
{
    if(species && species->isSetBoundaryCondition()) return species->getBoundaryCondition();
    return false;
}

int state::Species::setBoundaryCondition(int value)
{
    if(species) return species->setBoundaryCondition((bool)value);
    return -1;
}

int state::Species::getCharge() const
{
    if(species) return species->getCharge();
    return 0;
}

bool state::Species::getConstant() const
{
    if(species && species->isSetConstant()) return species->getConstant();
    return false;
}

int state::Species::setConstant(int value)
{
    if(species) return species->setBoundaryCondition((bool)value);
    return -1;
}

const std::string state::Species::getConversionFactor() const
{
    if(species) return species->getConversionFactor();
    return "";
}

bool state::Species::isSetId() const
{
    if(species) return species->isSetId();
    return false;
}

bool state::Species::isSetName() const
{
    if(species) return species->isSetName();
    return false;
}

bool state::Species::isSetSpeciesType() const
{
    if(species) return species->isSetSpeciesType();
    return false;
}

bool state::Species::isSetCompartment() const
{
    if(species) return species->isSetCompartment();
    return false;
}

bool state::Species::isSetInitialAmount() const
{
    if(species) return species->isSetInitialAmount();
    return false;
}

bool state::Species::isSetInitialConcentration() const
{
    if(species) return species->isSetInitialConcentration();
    return false;
}

bool state::Species::isSetSubstanceUnits() const
{
    if(species) return species->isSetSubstanceUnits();
    return false;
}

bool state::Species::isSetSpatialSizeUnits() const
{
    if(species) return species->isSetSpatialSizeUnits();
    return false;
}

bool state::Species::isSetUnits() const
{
    if(species) return species->isSetUnits();
    return false;
}

bool state::Species::isSetCharge() const
{
    if(species) return species->isSetCharge();
    return false;
}

bool state::Species::isSetConversionFactor() const
{
    if(species) return species->isSetConversionFactor();
    return false;
}

bool state::Species::isSetConstant() const
{
    if(species) return species->isSetConstant();
    return false;
}

bool state::Species::isSetBoundaryCondition() const
{
    if(species) return species->isSetBoundaryCondition();
    return false;
}

bool state::Species::isSetHasOnlySubstanceUnits() const
{
    if(species) return species->isSetHasOnlySubstanceUnits();
    return false;
}

int state::Species::setCompartment(const char *sid)
{
    if(species) return species->setCompartment(sid);
    return -1;
}

int state::Species::setSubstanceUnits(const char *sid)
{
    if(species) return species->setSubstanceUnits(sid);
    return -1;
}

int state::Species::setSpatialSizeUnits(const char *sid)
{
    if(species) return species->setSpatialSizeUnits(sid);
    return -1;
}

int state::Species::setUnits(const char *sname)
{
    if(species) return species->setUnits(sname);
    return -1;
}

int state::Species::setCharge(int value)
{
    if(species) return species->setCharge(value);
    return -1;
}

int state::Species::setConversionFactor(const char *sid)
{
    if(species) return species->setConversionFactor(sid);
    return -1;
}

int state::Species::unsetId()
{
    if(species) return species->unsetId();
    return -1;
}

int state::Species::unsetName()
{
    if(species) return species->unsetName();
    return -1;
}

int state::Species::unsetConstant()
{
    if(species) return species->unsetConstant();
    return -1;
}

int state::Species::unsetSpeciesType()
{
    if(species) return species->unsetSpeciesType();
    return -1;
}

int state::Species::unsetInitialAmount()
{
    if(species) return species->unsetInitialAmount();
    return -1;
}

int state::Species::unsetInitialConcentration()
{
    if(species) return species->unsetInitialConcentration();
    return -1;
}

int state::Species::unsetSubstanceUnits()
{
    if(species) return species->unsetSubstanceUnits();
    return -1;
}

int state::Species::unsetSpatialSizeUnits()
{
    if(species) return species->unsetSpatialSizeUnits();
    return -1;
}

int state::Species::unsetUnits()
{
    if(species) return species->unsetUnits();
    return -1;
}

int state::Species::unsetCharge()
{
    if(species) return species->unsetCharge();
    return -1;
}

int state::Species::unsetConversionFactor()
{
    if(species) return species->unsetConversionFactor();
    return -1;
}

int state::Species::unsetCompartment()
{
    if(species) return species->unsetCompartment();
    return -1;
}

int state::Species::unsetBoundaryCondition()
{
    if(species) return species->unsetBoundaryCondition();
    return -1;
}

int state::Species::unsetHasOnlySubstanceUnits()
{
    if(species) return species->unsetHasOnlySubstanceUnits();
    return -1;
}

int state::Species::hasRequiredAttributes()
{
    if(species) return species->hasRequiredAttributes();
    return -1;
}

static int species_init(state::Species *self, const std::string &s) {
    try {
        
        static std::regex e ("\\s*(const\\s+)?(\\$)?(\\w+)(\\s*=\\s*)?([-+]?[0-9]*\\.?[0-9]+)?\\s*");
        
        std::smatch sm;    // same as std::match_results<string::const_iterator> sm;
        
        // if we have a match, it looks like this:
        // matches for "const S1 = 234234.5"
        // match(0):(19)"const S1 = 234234.5"
        // match(1):(6)"const "
        // match(2):(0)""
        // match(3):(2)"S1"
        // match(4):(3)" = "
        // match(5):(8)"234234.5"
        static const int CNST = 1;  // Windows complains if name is CONST
        static const int BOUNDARY = 2;
        static const int ID = 3;
        static const int EQUAL = 4;
        static const int INIT = 5;
        
        if(std::regex_match (s,sm,e) && sm.size() == 6) {
            // check if name is valid sbml id
            if(!sm[ID].matched || !libsbml::SyntaxChecker_isValidSBMLSId(sm[ID].str().c_str())) {
                tf_exp(std::runtime_error("invalid Species id: \"" + sm[ID].str() + "\""));
                return -1;
            }
            
            if(sm[INIT].matched && !sm[EQUAL].matched) {
                tf_exp(std::runtime_error("Species has initial assignment value without equal symbol: \"" + s + "\""));
                return -1;
            }
            
            self->species = new libsbml::Species(state::getSBMLNamespaces());
            self->species->setId(sm[ID].str());
            self->species->setBoundaryCondition(sm[BOUNDARY].matched);
            self->species->setConstant(sm[CNST].matched);
            
            if(sm[INIT].matched) {
                self->species->setInitialConcentration(std::stod(sm[INIT].str()));
            }
            
            return 0;
        }
        else {
            tf_exp(std::runtime_error("invalid Species string: \"" + s + "\""));
            return -1;
        }
    }
    catch(const std::exception &e) {
        tf_exp(std::runtime_error("error creating Species(" + s + "\") : " + e.what()));
        return -1;
    }
    return -1;
}

std::string state::Species::str() const {
    std::string s = "Species('";
    if(species->isSetBoundaryCondition() && species->getBoundaryCondition()) {
        s += "$";
    }
    s += species->getId();
    s += "')";
    return s;
}

int32_t state::Species::flags() const
{
    int32_t r = 0;

    if(species) {
    
        if(species->getBoundaryCondition()) {
            r |= SPECIES_BOUNDARY;
        }
        
        if(species->getHasOnlySubstanceUnits()) {
            r |= SPECIES_SUBSTANCE;
        }
        
        if(species->getConstant()) {
            r |= SPECIES_CONSTANT;
        }

    }
    
    return r;
}

void state::Species::initDefaults()
{
    species->initDefaults();
}

state::Species::Species() {}

state::Species::Species(const std::string &s) : Species() {
    int result = species_init(this, s);
    if(result) TF_Log(LOG_CRITICAL) << "Species creation failed with return code " << result;
    else TF_Log(LOG_DEBUG) << "Species creation success: " << species->getId();
}

state::Species::Species(const state::Species &other) : Species() {
    species = libsbml::Species_clone(other.species);
}

state::Species::~Species() {
    if(species) {
        delete species;
        species = 0;
    }
}

std::string state::Species::toString() {
    return io::toString(*this);
}

state::Species *state::Species::fromString(const std::string &str) {
    return new state::Species(io::fromString<state::Species>(str));
}


namespace TissueForge::io { 

    template <>
    HRESULT toFile(const state::Species &dataElement, const MetaData &metaData, IOElement &fileElement) {
        TF_IOTOEASY(fileElement, metaData, "id", dataElement.getId());
        TF_IOTOEASY(fileElement, metaData, "name", dataElement.getName());
        TF_IOTOEASY(fileElement, metaData, "speciesType", dataElement.getSpeciesType());
        TF_IOTOEASY(fileElement, metaData, "compartment", dataElement.getCompartment());
        TF_IOTOEASY(fileElement, metaData, "initialAmount", dataElement.getInitialAmount());
        TF_IOTOEASY(fileElement, metaData, "initialConcentration", dataElement.getInitialConcentration());
        TF_IOTOEASY(fileElement, metaData, "substanceUnits", dataElement.getSubstanceUnits());
        TF_IOTOEASY(fileElement, metaData, "spatialSizeUnits", dataElement.getSpatialSizeUnits());
        TF_IOTOEASY(fileElement, metaData, "units", dataElement.getUnits());
        TF_IOTOEASY(fileElement, metaData, "hasOnlySubstanceUnits", dataElement.getHasOnlySubstanceUnits());
        TF_IOTOEASY(fileElement, metaData, "boundaryCondition", dataElement.getBoundaryCondition());
        TF_IOTOEASY(fileElement, metaData, "charge", dataElement.getCharge());
        TF_IOTOEASY(fileElement, metaData, "constant", dataElement.getConstant());
        TF_IOTOEASY(fileElement, metaData, "conversionFactor", dataElement.getConversionFactor());

        return S_OK;
    }

    template <>
    HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, state::Species *dataElement) {
        std::string s;
        FloatP_t d;
        int i;

        dataElement->species = new libsbml::Species(state::getSBMLNamespaces());

        TF_IOFROMEASY(fileElement, metaData, "id", &s);
        dataElement->setId(s.c_str());

        TF_IOFROMEASY(fileElement, metaData, "name", &s);
        dataElement->setName(s.c_str());

        TF_IOFROMEASY(fileElement, metaData, "speciesType", &s);
        dataElement->setSpeciesType(s.c_str());

        TF_IOFROMEASY(fileElement, metaData, "compartment", &s);
        dataElement->setCompartment(s.c_str());

        TF_IOFROMEASY(fileElement, metaData, "initialAmount", &d);
        dataElement->setInitialAmount(d);

        TF_IOFROMEASY(fileElement, metaData, "initialConcentration", &d);
        dataElement->setInitialConcentration(d);

        TF_IOFROMEASY(fileElement, metaData, "substanceUnits", &s);
        dataElement->setSubstanceUnits(s.c_str());

        TF_IOFROMEASY(fileElement, metaData, "spatialSizeUnits", &s);
        dataElement->setSpatialSizeUnits(s.c_str());

        TF_IOFROMEASY(fileElement, metaData, "units", &s);
        dataElement->setUnits(s.c_str());

        TF_IOFROMEASY(fileElement, metaData, "hasOnlySubstanceUnits", &i);
        dataElement->setHasOnlySubstanceUnits(i);

        TF_IOFROMEASY(fileElement, metaData, "boundaryCondition", &i);
        dataElement->setBoundaryCondition(i);

        TF_IOFROMEASY(fileElement, metaData, "charge", &i);
        dataElement->setCharge(i);

        TF_IOFROMEASY(fileElement, metaData, "constant", &i);
        dataElement->setConstant(i);

        TF_IOFROMEASY(fileElement, metaData, "conversionFactor", &s);
        dataElement->setConversionFactor(s.c_str());

        return S_OK;
    }

};
