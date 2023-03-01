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

#include "tfCTest.h"


int main(int argc, char** argv) {
    //  dimensions of universe
    tfFloatP_t dim[] = {10.0, 10.0, 10.0};

    int cells[] = {5, 5, 5};
    tfFloatP_t cutoff = 1.0;

    // new simulator
    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;
    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfUniverseConfig_setDim(&uconfig, dim));
    TFC_TEST_CHECK(tfUniverseConfig_setCells(&uconfig, cells));
    TFC_TEST_CHECK(tfUniverseConfig_setCutoff(&uconfig, cutoff));
    TFC_TEST_CHECK(tfInitC(&config, NULL, 0));

    // create a potential representing a 12-6 Lennard-Jones potential
    struct tfPotentialHandle pot;
    tfFloatP_t pottol = 1.0e-3;
    TFC_TEST_CHECK(tfPotential_create_lennard_jones_12_6(&pot, 0.275, 1.0, 9.5075e-06, 6.1545e-03, &pottol));

    // create a particle type
    struct tfParticleTypeHandle Argon;
    TFC_TEST_CHECK(tfParticleType_init(&Argon));
    TFC_TEST_CHECK(tfParticleType_setName(&Argon, "ArgonType"));
    TFC_TEST_CHECK(tfParticleType_setRadius(&Argon, 0.1));
    TFC_TEST_CHECK(tfParticleType_setMass(&Argon, 39.4));
    TFC_TEST_CHECK(tfParticleType_registerType(&Argon));

    // bind the potential with the *TYPES* of the particles
    TFC_TEST_CHECK(tfBindTypes(&pot, &Argon, &Argon, 0));

    // uniform cube
    TFC_TEST_CHECK(tfParticleType_factory(&Argon, NULL, 2500, NULL, NULL));

    // run the simulator
    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}