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
    tfFloatP_t dim[] = {30.0, 30.0, 30.0};
    int cells[] = {3, 3, 3};
    tfFloatP_t dist = 3.9;
    tfFloatP_t offset = 6.0;
    tfFloatP_t dt = 0.01;

    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;

    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));

    TFC_TEST_CHECK(tfUniverseConfig_setDim(&uconfig, dim));
    TFC_TEST_CHECK(tfUniverseConfig_setCutoff(&uconfig, 7.0));
    TFC_TEST_CHECK(tfUniverseConfig_setCells(&uconfig, cells));
    TFC_TEST_CHECK(tfUniverseConfig_setDt(&uconfig, dt));

    TFC_TEST_CHECK(tfInitC(&config, NULL, 0));

    struct tfParticleTypeSpec ATypeDef = tfParticleTypeSpec_init();
    struct tfParticleTypeSpec SphereTypeDef = tfParticleTypeSpec_init();
    struct tfParticleTypeSpec TestTypeDef = tfParticleTypeSpec_init();
    struct tfParticleTypeStyleSpec AStyleDef = tfParticleTypeStyleSpec_init();
    struct tfParticleTypeStyleSpec SphereStyleDef = tfParticleTypeStyleSpec_init();
    struct tfParticleTypeStyleSpec TestStyleDef = tfParticleTypeStyleSpec_init();

    AStyleDef.color = "MediumSeaGreen";
    ATypeDef.mass = 2.5;
    ATypeDef.style = &AStyleDef;

    SphereStyleDef.color = "orange";
    SphereTypeDef.radius = 3.0;
    SphereTypeDef.frozen = 1;
    SphereTypeDef.style = &SphereStyleDef;

    TestStyleDef.color = "orange";
    TestTypeDef.radius = 0.0;
    TestTypeDef.frozen = 1;
    TestTypeDef.style = &TestStyleDef;

    struct tfParticleTypeHandle AType, SphereType, TestType;
    TFC_TEST_CHECK(tfParticleType_initD(&AType, ATypeDef));
    TFC_TEST_CHECK(tfParticleType_initD(&SphereType, SphereTypeDef));
    TFC_TEST_CHECK(tfParticleType_initD(&TestType, TestTypeDef));

    TFC_TEST_CHECK(tfParticleType_registerType(&AType));
    TFC_TEST_CHECK(tfParticleType_registerType(&SphereType));
    TFC_TEST_CHECK(tfParticleType_registerType(&TestType));

    struct tfPotentialHandle p;
    tfFloatP_t p_d = 100.0;
    tfFloatP_t p_a = 1.0;
    tfFloatP_t p_min = -3.0;
    tfFloatP_t p_max = 4.0;
    TFC_TEST_CHECK(tfPotential_create_morse(&p, &p_d, &p_a, NULL, &p_min, &p_max, NULL));

    TFC_TEST_CHECK(tfBindTypes(&p, &AType, &SphereType, 0));
    TFC_TEST_CHECK(tfBindTypes(&p, &AType, &TestType, 0));

    struct tfBoundaryConditionsHandle bcs;
    TFC_TEST_CHECK(tfUniverse_getBoundaryConditions(&bcs));
    TFC_TEST_CHECK(tfBoundaryConditions_setPotential(&bcs, &AType, &p));

    tfFloatP_t *center = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    TFC_TEST_CHECK(tfUniverse_getCenter(&center));
    tfFloatP_t pos[] = {5.0 + center[0], center[1], center[2]};
    int pid;

    TFC_TEST_CHECK(tfParticleType_createParticle(&SphereType, &pid, pos, NULL));
    
    pos[2] = center[2] + dist;
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[0] = center[0] + 6.0;
    pos[1] = center[1] - 6.0;
    pos[2] = center[2] + 6.0;
    TFC_TEST_CHECK(tfParticleType_createParticle(&TestType, &pid, pos, NULL));
    
    pos[2] += dist;
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[0] = center[0];
    pos[1] = center[1];
    pos[2] = dist;
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[2] = dim[2] - dist;
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[0] = dist;
    pos[1] = center[1] - offset;
    pos[2] = center[2];
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[0] = dim[0] - dist;
    pos[1] = center[1] + offset;
    pos[2] = center[2];
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[0] = center[0];
    pos[1] = dist;
    pos[2] = center[2];
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, pos, NULL));

    pos[1] = dim[1] - dist;
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, pos, NULL));

    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}