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

#include "tfSpeciesValue.h"
#include <state/tfStateVector.h>
#include "tfSpeciesList.h"
#include <tfError.h>

#include <tfSecreteUptake.h>

#include <sbml/Species.h>
#include <iostream>


using namespace TissueForge;


state::Species *state::SpeciesValue::species() {
    if(!(state_vector && state_vector->species)) return NULL;
    return state_vector->species->item(index);
}

state::SpeciesValue::SpeciesValue(struct state::StateVector *state_vector, uint32_t index) : 
    state_vector(state_vector), index(index) 
{}

FloatP_t state::SpeciesValue::getValue() const {
    if(!state_vector) return -FPTYPE_ONE;
    return state_vector->fvec[index];
}

void state::SpeciesValue::setValue(const FloatP_t &_value) {
    if(state_vector) 
        state_vector->fvec[index] = FPTYPE_FMAX(FPTYPE_ZERO, _value);
}

bool state::SpeciesValue::getBoundaryCondition() {
    return species()->getBoundaryCondition();
}

int state::SpeciesValue::setBoundaryCondition(const int &value) {
    return species()->setBoundaryCondition(value);
}

FloatP_t state::SpeciesValue::getInitialAmount() {
    return species()->getInitialAmount();
}

int state::SpeciesValue::setInitialAmount(const FloatP_t &value) {
    return species()->setInitialAmount(value);
}

FloatP_t state::SpeciesValue::getInitialConcentration() {
    return species()->getInitialConcentration();
}

int state::SpeciesValue::setInitialConcentration(const FloatP_t &value) {
    return species()->setInitialConcentration(value);
}

bool state::SpeciesValue::getConstant() {
    return species()->getConstant();
}

int state::SpeciesValue::setConstant(const int &value) {
    return species()->setConstant(value);
}

FloatP_t state::SpeciesValue::secrete(const FloatP_t &amount, const ParticleList &to) {
    return SecreteUptake::secrete(this, amount, to);
}

FloatP_t state::SpeciesValue::secrete(const FloatP_t &amount, const FloatP_t &distance) {
    return SecreteUptake::secrete(this, amount, distance);
}
