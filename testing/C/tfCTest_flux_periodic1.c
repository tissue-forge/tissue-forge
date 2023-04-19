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
    tfFloatP_t dim[] = {15.0, 6.0, 6.0};
    int cells[] = {9, 3, 3};
    
    struct tfBoundaryConditionKindHandle bcKindEnum;
    TFC_TEST_CHECK(tfBoundaryConditionKind_init(&bcKindEnum));

    struct tfBoundaryConditionsArgsContainerHandle bargs;
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_init(&bargs));
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_setValue(&bargs, "x", bcKindEnum.BOUNDARY_RESETTING));

    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;

    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfUniverseConfig_setDim(&uconfig, dim));
    TFC_TEST_CHECK(tfUniverseConfig_setDt(&uconfig, 0.1));
    TFC_TEST_CHECK(tfUniverseConfig_setCutoff(&uconfig, 3.0));
    TFC_TEST_CHECK(tfUniverseConfig_setNumFluxSteps(&uconfig, 2));
    TFC_TEST_CHECK(tfUniverseConfig_setBoundaryConditions(&uconfig, &bargs));
    TFC_TEST_CHECK(tfInitC(&config, NULL, 0));

    struct tfParticleTypeStyleSpec ATypeStyleDef = tfParticleTypeStyleSpec_init();
    struct tfParticleTypeSpec ATypeDef = tfParticleTypeSpec_init();

    ATypeStyleDef.speciesName = "S1";
    ATypeDef.numSpecies = 3;
    ATypeDef.species = (char**)malloc(3 * sizeof(char*));
    ATypeDef.species[0] = "S1";
    ATypeDef.species[1] = "S2";
    ATypeDef.species[2] = "S3";
    ATypeDef.style = &ATypeStyleDef;

    struct tfParticleTypeHandle AType;
    TFC_TEST_CHECK(tfParticleType_initD(&AType, ATypeDef));
    TFC_TEST_CHECK(tfParticleType_registerType(&AType));

    struct tfFluxesHandle flux;
    TFC_TEST_CHECK(tfFluxes_fluxFick(&flux, &AType, &AType, "S1", 2.0, 0.0));

    tfFloatP_t *center = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    TFC_TEST_CHECK(tfUniverse_getCenter(&center));

    tfFloatP_t pos[] = {center[0], center[1] - 1.0, center[2]};
    int pid0, pid1;
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid0, pos, NULL));
    pos[0] = center[0] - 5.0;
    pos[1] = center[1] + 1.0;
    tfFloatP_t velocity[] = {0.5, 0.0, 0.0};
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid1, pos, velocity));

    struct tfParticleHandleHandle part0, part1;
    TFC_TEST_CHECK(tfParticleHandle_init(&part0, pid0));
    TFC_TEST_CHECK(tfParticleHandle_init(&part1, pid1));

    struct tfStateStateVectorHandle svec0, svec1;
    TFC_TEST_CHECK(tfParticleHandle_getSpecies(&part0, &svec0));
    TFC_TEST_CHECK(tfParticleHandle_getSpecies(&part1, &svec1));

    TFC_TEST_CHECK(tfStateStateVector_setItem(&svec0, 0, 1.0));
    TFC_TEST_CHECK(tfStateStateVector_setItem(&svec1, 0, 0.0));

    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}