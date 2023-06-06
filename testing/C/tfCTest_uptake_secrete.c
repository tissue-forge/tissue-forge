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
    tfFloatP_t dim[] = {6.5, 6.5, 6.5};

    struct tfBoundaryConditionSpaceKindHandle bcEnums;
    TFC_TEST_CHECK(tfBoundaryConditionSpaceKind_init(&bcEnums));

    struct tfBoundaryConditionsArgsContainerHandle bargs;
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_init(&bargs));
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_setValueAll(&bargs, bcEnums.SPACE_FREESLIP_FULL));

    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;
    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfUniverseConfig_setDim(&uconfig, dim));
    TFC_TEST_CHECK(tfUniverseConfig_setBoundaryConditions(&uconfig, &bargs));
    TFC_TEST_CHECK(tfInitC(&config, NULL, 0));

    struct tfParticleTypeStyleSpec ATypeStyleDef = tfParticleTypeStyleSpec_init();
    struct tfParticleTypeStyleSpec ProducerTypeStyleDef = tfParticleTypeStyleSpec_init();
    struct tfParticleTypeStyleSpec ConsumerTypeStyleDef = tfParticleTypeStyleSpec_init();

    struct tfParticleTypeSpec ATypeDef = tfParticleTypeSpec_init();
    struct tfParticleTypeSpec ProducerTypeDef = tfParticleTypeSpec_init();
    struct tfParticleTypeSpec ConsumerTypeDef = tfParticleTypeSpec_init();

    ATypeDef.radius = 0.1;
    ATypeDef.numSpecies = 3;
    ATypeDef.species = (char**)malloc(3 * sizeof(char*));
    ATypeDef.species[0] = "S1";
    ATypeDef.species[1] = "S2";
    ATypeDef.species[2] = "S3";
    ATypeStyleDef.speciesName = "S1";
    ATypeDef.style = &ATypeStyleDef;

    ProducerTypeDef.radius = 0.1;
    ProducerTypeDef.numSpecies = 3;
    ProducerTypeDef.species = (char**)malloc(3 * sizeof(char*));
    ProducerTypeDef.species[0] = "S1";
    ProducerTypeDef.species[1] = "S2";
    ProducerTypeDef.species[2] = "S3";
    ProducerTypeStyleDef.speciesName = "S1";
    ProducerTypeDef.style = &ProducerTypeStyleDef;

    ConsumerTypeDef.radius = 0.1;
    ConsumerTypeDef.numSpecies = 3;
    ConsumerTypeDef.species = (char**)malloc(3 * sizeof(char*));
    ConsumerTypeDef.species[0] = "S1";
    ConsumerTypeDef.species[1] = "S2";
    ConsumerTypeDef.species[2] = "S3";
    ConsumerTypeStyleDef.speciesName = "S1";
    ConsumerTypeDef.style = &ConsumerTypeStyleDef;

    struct tfParticleTypeHandle AType, ProducerType, ConsumerType;
    TFC_TEST_CHECK(tfParticleType_initD(&AType, ATypeDef));
    TFC_TEST_CHECK(tfParticleType_initD(&ProducerType, ProducerTypeDef));
    TFC_TEST_CHECK(tfParticleType_initD(&ConsumerType, ConsumerTypeDef));
    TFC_TEST_CHECK(tfParticleType_registerType(&AType));
    TFC_TEST_CHECK(tfParticleType_registerType(&ProducerType));
    TFC_TEST_CHECK(tfParticleType_registerType(&ConsumerType));

    struct tfFluxesHandle fluxAA, fluxPA, fluxAC;
    TFC_TEST_CHECK(tfFluxes_fluxFick(&fluxAA, &AType, &AType, "S1", 1.0, 0.0, -1));
    TFC_TEST_CHECK(tfFluxes_fluxFick(&fluxPA, &ProducerType, &AType, "S1", 1.0, 0.0, -1));
    TFC_TEST_CHECK(tfFluxes_fluxFick(&fluxAC, &AType, &ConsumerType, "S1", 2.0, 10.0, -1));

    tfFloatP_t *posP = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    tfFloatP_t *posC = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    tfFloatP_t offset = 1.0;
    tfFloatP_t *center = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    TFC_TEST_CHECK(tfUniverse_getCenter(&center));
    posP[0] = offset; posP[1] = center[1]; posP[2] = center[2];
    posC[0] = dim[0] - offset; posC[1] = center[1]; posC[2] = center[2];

    int pid;
    TFC_TEST_CHECK(tfParticleType_createParticle(&ProducerType, &pid, posP, NULL));
    TFC_TEST_CHECK(tfParticleType_createParticle(&ConsumerType, &pid, posC, NULL));

    struct tfParticleHandleHandle partP;
    TFC_TEST_CHECK(tfParticleType_getParticle(&ProducerType, 0, &partP));
    struct tfStateStateVectorHandle svec;
    TFC_TEST_CHECK(tfParticleHandle_getSpecies(&partP, &svec));
    TFC_TEST_CHECK(tfStateStateVector_setItem(&svec, 0, 200.0));

    for(unsigned int i = 0; i < 1000; i++) 
        TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, NULL, NULL));

    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}