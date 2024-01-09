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

    int A_count = 5000;
    int B_count = 5000;
    tfFloatP_t dim[] = {20.0, 20.0, 20.0};
    tfFloatP_t cutoff = 3.0;

    struct tfSimulatorConfigHandle config;
    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    struct tfUniverseConfigHandle uconfig;
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfUniverseConfig_setDim(&uconfig, dim));
    TFC_TEST_CHECK(tfUniverseConfig_setCutoff(&uconfig, cutoff));
    TFC_TEST_CHECK(tfTest_initC(&config));

    tfFloatP_t dt;
    TFC_TEST_CHECK(tfUniverse_getDt(&dt));

    struct tfParticleDynamicsEnumHandle dynEnums;
    TFC_TEST_CHECK(tfParticleDynamics_init(&dynEnums));

    struct tfRenderingStyleHandle AStyle, BStyle;
    TFC_TEST_CHECK(tfRenderingStyle_init(&AStyle));
    TFC_TEST_CHECK(tfRenderingStyle_setColor(&AStyle, "red"));
    
    TFC_TEST_CHECK(tfRenderingStyle_init(&BStyle));
    TFC_TEST_CHECK(tfRenderingStyle_setColor(&BStyle, "blue"));

    struct tfParticleTypeHandle AType, BType;

    TFC_TEST_CHECK(tfParticleType_init(&AType));
    TFC_TEST_CHECK(tfParticleType_setMass(&AType, 40.0));
    TFC_TEST_CHECK(tfParticleType_setRadius(&AType, 0.4));
    TFC_TEST_CHECK(tfParticleType_setDynamics(&AType, dynEnums.PARTICLE_OVERDAMPED));
    TFC_TEST_CHECK(tfParticleType_setStyle(&AType, &AStyle));

    TFC_TEST_CHECK(tfParticleType_init(&BType));
    TFC_TEST_CHECK(tfParticleType_setMass(&BType, 40.0));
    TFC_TEST_CHECK(tfParticleType_setRadius(&BType, 0.4));
    TFC_TEST_CHECK(tfParticleType_setDynamics(&BType, dynEnums.PARTICLE_OVERDAMPED));
    TFC_TEST_CHECK(tfParticleType_setStyle(&BType, &BStyle));

    TFC_TEST_CHECK(tfParticleType_registerType(&AType));
    TFC_TEST_CHECK(tfParticleType_registerType(&BType));

    struct tfPotentialHandle pot_aa, pot_bb, pot_ab;

    tfFloatP_t aa_d = 3.0, aa_a = 5.0, aa_max = 3.0;
    tfFloatP_t bb_d = 3.0, bb_a = 5.0, bb_max = 3.0;
    tfFloatP_t ab_d = 0.3, ab_a = 5.0, ab_max = 3.0;

    TFC_TEST_CHECK(tfPotential_create_morse(&pot_aa, &aa_d, &aa_a, NULL, NULL, &aa_max, NULL));
    TFC_TEST_CHECK(tfPotential_create_morse(&pot_bb, &bb_d, &bb_a, NULL, NULL, &bb_max, NULL));
    TFC_TEST_CHECK(tfPotential_create_morse(&pot_ab, &ab_d, &ab_a, NULL, NULL, &ab_max, NULL));

    TFC_TEST_CHECK(tfBindTypes(&pot_aa, &AType, &AType, 0));
    TFC_TEST_CHECK(tfBindTypes(&pot_bb, &BType, &BType, 0));
    TFC_TEST_CHECK(tfBindTypes(&pot_ab, &AType, &BType, 0));

    struct tfGaussianHandle force_rnd;
    struct tfForceHandle force_rnd_base;
    TFC_TEST_CHECK(tfGaussian_init(&force_rnd, 50.0, 0.0, dt));
    TFC_TEST_CHECK(tfGaussian_toBase(&force_rnd, &force_rnd_base));
    TFC_TEST_CHECK(tfBindForce(&force_rnd_base, &AType));
    TFC_TEST_CHECK(tfBindForce(&force_rnd_base, &BType));

    struct tfPointsTypeHandle ptTypeEnums;
    TFC_TEST_CHECK(tfPointsType_init(&ptTypeEnums));
    tfFloatP_t *ptsA, *ptsB;
    TFC_TEST_CHECK(tfRandomPoints(ptTypeEnums.SolidCube, A_count, 0.0, 0.0, 0.0, &ptsA));
    TFC_TEST_CHECK(tfRandomPoints(ptTypeEnums.SolidCube, B_count, 0.0, 0.0, 0.0, &ptsB));

    tfFloatP_t *center = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    TFC_TEST_CHECK(tfUniverse_getCenter(&center));

    int pid;
    tfFloatP_t ptp[3];
    for(unsigned int i = 0; i < A_count; i++) {
        for(unsigned int j = 0; j < 3; j++) 
            ptp[j] = center[j] + ptsA[3 * i + j] * 14.5;
        TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, ptp, NULL));
    }
    for(unsigned int i = 0; i < B_count; i++) {
        for(unsigned int j = 0; j < 3; j++) 
            ptp[j] = center[j] + ptsB[3 * i + j] * 14.5;
        TFC_TEST_CHECK(tfParticleType_createParticle(&BType, &pid, ptp, NULL));
    }

    TFC_TEST_CHECK(tfTest_runQuiet(10));

    return 0;
}