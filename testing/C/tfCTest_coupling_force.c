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

#include "tfCTest.h"


int main(int argc, char** argv) {
    tfFloatP_t dim[] = {6.5, 6.5, 6.5};

    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;
    struct tfBoundaryConditionsArgsContainerHandle bargs;
    struct tfBoundaryConditionSpaceKindHandle bcKindEnum;

    TFC_TEST_CHECK(tfBoundaryConditionSpaceKind_init(&bcKindEnum));
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_init(&bargs));
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_setValueAll(&bargs, bcKindEnum.SPACE_FREESLIP_FULL));

    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfUniverseConfig_setBoundaryConditions(&uconfig, &bargs));
    TFC_TEST_CHECK(tfUniverseConfig_setDim(&uconfig, dim));
    TFC_TEST_CHECK(tfTest_initC(&config));

    struct tfParticleTypeSpec ATypeDef = tfParticleTypeSpec_init();
    struct tfParticleTypeSpec BTypeDef = tfParticleTypeSpec_init();
    ATypeDef.radius = 0.1;
    ATypeDef.species = (char**)malloc(sizeof(char*));
    ATypeDef.species[0] = "S1";
    ATypeDef.numSpecies = 1;
    BTypeDef.species = (char**)malloc(sizeof(char*));
    BTypeDef.species[0] = "S1";
    BTypeDef.numSpecies = 1;

    struct tfParticleTypeHandle AType, BType;
    TFC_TEST_CHECK(tfParticleType_initD(&AType, ATypeDef));
    TFC_TEST_CHECK(tfParticleType_registerType(&AType));
    TFC_TEST_CHECK(tfParticleType_initD(&BType, BTypeDef));
    TFC_TEST_CHECK(tfParticleType_registerType(&BType));

    struct tfStateSpeciesListHandle slistB;
    TFC_TEST_CHECK(tfParticleType_getSpecies(&BType, &slistB));
    struct tfStateSpeciesHandle S1B;
    TFC_TEST_CHECK(tfStateSpeciesList_getItemS(&slistB, "S1", &S1B));
    TFC_TEST_CHECK(tfStateSpecies_setConstant(&S1B, 1));

    tfFloatP_t dt;
    TFC_TEST_CHECK(tfUniverse_getDt(&dt));

    struct tfGaussianHandle force_rnd;
    struct tfForceHandle force_rnd_base;
    TFC_TEST_CHECK(tfGaussian_init(&force_rnd, 0.1, 1.0, dt));
    TFC_TEST_CHECK(tfGaussian_toBase(&force_rnd, &force_rnd_base));

    TFC_TEST_CHECK(tfBindForceS(&force_rnd_base, &AType, "S1"));
    TFC_TEST_CHECK(tfBindForceS(&force_rnd_base, &BType, "S1"));

    struct tfFluxesHandle fluxhAA, fluxhAB;
    TFC_TEST_CHECK(tfFluxes_fluxFick(&fluxhAA, &AType, &AType, "S1", 1.0, 0.0, -1));
    TFC_TEST_CHECK(tfFluxes_fluxFick(&fluxhAB, &AType, &BType, "S1", 1.0, 0.0, -1));

    unsigned int numParts = 500;
    int pid;
    for(unsigned int i = 0; i < numParts; i++) {
        TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, NULL, NULL));
    }

    struct tfParticleHandleHandle part;
    TFC_TEST_CHECK(tfParticleType_getParticle(&AType, 0, &part));
    TFC_TEST_CHECK(tfParticleHandle_become(&part, &BType));

    struct tfStateStateVectorHandle svec;
    TFC_TEST_CHECK(tfParticleHandle_getSpecies(&part, &svec));
    TFC_TEST_CHECK(tfStateStateVector_setItem(&svec, 0, 10.0));

    TFC_TEST_CHECK(tfTest_runQuiet(20));

    return 0;
}