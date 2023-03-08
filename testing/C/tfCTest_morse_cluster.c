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
    tfFloatP_t cutoff = 10.0;
    tfFloatP_t dt = 0.0005;

    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;

    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfUniverseConfig_setDim(&uconfig, dim));
    TFC_TEST_CHECK(tfUniverseConfig_setCutoff(&uconfig, cutoff));
    TFC_TEST_CHECK(tfUniverseConfig_setDt(&uconfig, dt));
    TFC_TEST_CHECK(tfInitC(&config, NULL, 0));

    struct tfParticleTypeSpec ATypeDef = tfParticleTypeSpec_init();
    struct tfParticleTypeSpec BTypeDef = tfParticleTypeSpec_init();
    struct tfParticleTypeStyleSpec ATypeStyleDef = tfParticleTypeStyleSpec_init();
    struct tfParticleTypeStyleSpec BTypeStyleDef = tfParticleTypeStyleSpec_init();

    ATypeDef.radius = 0.5;
    ATypeDef.dynamics = 1;
    ATypeDef.mass = 10.0;
    ATypeStyleDef.color = "MediumSeaGreen";
    ATypeDef.style = &ATypeStyleDef;

    BTypeDef.radius = 0.5;
    BTypeDef.dynamics = 1;
    BTypeDef.mass = 10.0;
    BTypeStyleDef.color = "skyblue";
    BTypeDef.style = &BTypeStyleDef;

    struct tfParticleTypeHandle AType, BType;
    TFC_TEST_CHECK(tfParticleType_initD(&AType, ATypeDef));
    TFC_TEST_CHECK(tfParticleType_initD(&BType, BTypeDef));
    TFC_TEST_CHECK(tfParticleType_registerType(&AType));
    TFC_TEST_CHECK(tfParticleType_registerType(&BType));

    struct tfClusterTypeSpec CTypeDef = tfClusterTypeSpec_init();
    CTypeDef.numTypes = 2;
    CTypeDef.types = (struct tfParticleTypeHandle**)malloc(CTypeDef.numTypes * sizeof(struct tfParticleTypeHandle*));
    CTypeDef.types[0] = &AType;
    CTypeDef.types[1] = &BType;

    struct tfClusterParticleTypeHandle CType;
    TFC_TEST_CHECK(tfClusterParticleType_initD(&CType, CTypeDef));
    TFC_TEST_CHECK(tfClusterParticleType_registerType(&CType));

    struct tfPotentialHandle p1, p2;
    tfFloatP_t p1_d = 0.5;
    tfFloatP_t p1_a = 5.0;
    tfFloatP_t p1_max = 3.0;
    tfFloatP_t p2_d = 0.5;
    tfFloatP_t p2_a = 2.5;
    tfFloatP_t p2_max = 3.0;
    TFC_TEST_CHECK(tfPotential_create_morse(&p1, &p1_d, &p1_a, NULL, NULL, &p1_max, NULL));
    TFC_TEST_CHECK(tfPotential_create_morse(&p2, &p2_d, &p2_a, NULL, NULL, &p2_max, NULL));
    TFC_TEST_CHECK(tfBindTypes(&p1, &AType, &AType, 1));
    TFC_TEST_CHECK(tfBindTypes(&p2, &BType, &BType, 1));

    struct tfGaussianHandle force;
    struct tfForceHandle force_base;
    TFC_TEST_CHECK(tfGaussian_init(&force, 10.0, 0.0, dt));
    TFC_TEST_CHECK(tfGaussian_toBase(&force, &force_base));
    TFC_TEST_CHECK(tfBindForce(&force_base, &AType));
    TFC_TEST_CHECK(tfBindForce(&force_base, &BType));

    tfFloatP_t *center = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    TFC_TEST_CHECK(tfUniverse_getCenter(&center));

    struct tfClusterParticleHandleHandle c1, c2;
    tfFloatP_t pos[] = {center[0] - 3.0, center[1], center[2]};
    int cid1, cid2;
    TFC_TEST_CHECK(tfClusterParticleType_createParticle(&CType, &cid1, pos, NULL));
    TFC_TEST_CHECK(tfClusterParticleHandle_init(&c1, cid1));

    pos[0] = center[0] + 7.0;
    TFC_TEST_CHECK(tfClusterParticleType_createParticle(&CType, &cid2, pos, NULL));
    TFC_TEST_CHECK(tfClusterParticleHandle_init(&c2, cid2));

    int pid;
    for(unsigned int i = 0; i < 2000; i++) {
        TFC_TEST_CHECK(tfClusterParticleHandle_createParticle(&c1, &AType, &pid, NULL, NULL));
        TFC_TEST_CHECK(tfClusterParticleHandle_createParticle(&c2, &BType, &pid, NULL, NULL));
    }

    TFC_TEST_CHECK(tfTest_runQuiet(20));

    return 0;
}