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

#include "tfCTest.h"


int main(int argc, char** argv) {
    //  dimensions of universe
    tfFloatP_t dim[] = {30.0, 30.0, 30.0};

    tfFloatP_t cutoff = 5.0;
    tfFloatP_t dt = 0.0005;

    // new simulator
    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;
    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfUniverseConfig_setDim(&uconfig, dim));
    TFC_TEST_CHECK(tfUniverseConfig_setCutoff(&uconfig, cutoff));
    TFC_TEST_CHECK(tfUniverseConfig_setDt(&uconfig, dt));
    TFC_TEST_CHECK(tfInitC(&config, NULL, 0));

    struct tfRenderingStyleHandle AStyle, BStyle, CStyle;
    TFC_TEST_CHECK(tfRenderingStyle_init(&AStyle));
    TFC_TEST_CHECK(tfRenderingStyle_init(&BStyle));
    TFC_TEST_CHECK(tfRenderingStyle_init(&CStyle));
    TFC_TEST_CHECK(tfRenderingStyle_setColor(&AStyle, "MediumSeaGreen"));
    TFC_TEST_CHECK(tfRenderingStyle_setColor(&BStyle, "skyblue"));
    TFC_TEST_CHECK(tfRenderingStyle_setColor(&CStyle, "orange"));

    tfFloatP_t Aradius = 0.5;
    tfFloatP_t Bradius = 0.2;
    tfFloatP_t Cradius = 10.0;

    struct tfParticleTypeHandle AType, BType, CType;
    struct tfParticleDynamicsEnumHandle dynEnums;
    TFC_TEST_CHECK(tfParticleType_init(&AType));
    TFC_TEST_CHECK(tfParticleType_init(&BType));
    TFC_TEST_CHECK(tfParticleType_init(&CType));
    TFC_TEST_CHECK(tfParticleDynamics_init(&dynEnums));

    TFC_TEST_CHECK(tfParticleType_setName(&AType, "AType"));
    TFC_TEST_CHECK(tfParticleType_setRadius(&AType, Aradius));
    TFC_TEST_CHECK(tfParticleType_setDynamics(&AType, dynEnums.PARTICLE_OVERDAMPED));
    TFC_TEST_CHECK(tfParticleType_setMass(&AType, 5.0));
    TFC_TEST_CHECK(tfParticleType_setStyle(&AType, &AStyle));

    TFC_TEST_CHECK(tfParticleType_setName(&BType, "BType"));
    TFC_TEST_CHECK(tfParticleType_setRadius(&BType, Bradius));
    TFC_TEST_CHECK(tfParticleType_setDynamics(&BType, dynEnums.PARTICLE_OVERDAMPED));
    TFC_TEST_CHECK(tfParticleType_setMass(&BType, 1.0));
    TFC_TEST_CHECK(tfParticleType_setStyle(&BType, &BStyle));

    TFC_TEST_CHECK(tfParticleType_setName(&CType, "CType"));
    TFC_TEST_CHECK(tfParticleType_setRadius(&CType, Cradius));
    TFC_TEST_CHECK(tfParticleType_setFrozen(&CType, 1));
    TFC_TEST_CHECK(tfParticleType_setStyle(&CType, &CStyle));

    TFC_TEST_CHECK(tfParticleType_registerType(&AType));
    TFC_TEST_CHECK(tfParticleType_registerType(&BType));
    TFC_TEST_CHECK(tfParticleType_registerType(&CType));

    tfFloatP_t *center = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    TFC_TEST_CHECK(tfUniverse_getCenter(&center));

    int pid;
    TFC_TEST_CHECK(tfParticleType_createParticle(&CType, &pid, center, NULL));

    struct tfPointsTypeHandle pointTypesEnum;
    TFC_TEST_CHECK(tfPointsType_init(&pointTypesEnum));

    tfFloatP_t *ringPoints;
    int numRingPoints = 100;
    TFC_TEST_CHECK(tfPoints(pointTypesEnum.Ring, numRingPoints, &ringPoints));

    tfFloatP_t pt[3];
    tfFloatP_t offset[] = {0.0, 0.0, -1.0};
    
    for(unsigned int i = 0; i < numRingPoints; i++) {
        for(unsigned int j = 0; j < 3; j++) {
            pt[j] = ringPoints[3 * i + j] * (Bradius + Cradius) + center[j] + offset[j];
        }
        TFC_TEST_CHECK(tfParticleType_createParticle(&BType, &pid, pt, NULL));
    }

    struct tfPotentialHandle pc, pa, pb, pab, ph;
    tfFloatP_t pc_m = 2.0;
    tfFloatP_t pc_max = 5.0;
    TFC_TEST_CHECK(tfPotential_create_glj(&pc, 30.0, &pc_m, NULL, NULL, NULL, NULL, &pc_max, NULL, NULL));
    tfFloatP_t pa_m = 2.5;
    tfFloatP_t pa_max = 3.0;
    TFC_TEST_CHECK(tfPotential_create_glj(&pa, 3.0, &pa_m, NULL, NULL, NULL, NULL, &pa_max, NULL, NULL));
    tfFloatP_t pb_m = 4.0;
    tfFloatP_t pb_max = 1.0;
    TFC_TEST_CHECK(tfPotential_create_glj(&pb, 1.0, &pb_m, NULL, NULL, NULL, NULL, &pb_max, NULL, NULL));
    tfFloatP_t pab_m = 2.0;
    tfFloatP_t pab_max = 1.0;
    TFC_TEST_CHECK(tfPotential_create_glj(&pab, 1.0, &pab_m, NULL, NULL, NULL, NULL, &pab_max, NULL, NULL));
    tfFloatP_t ph_r0 = 0.001;
    tfFloatP_t ph_k = 200.0;
    TFC_TEST_CHECK(tfPotential_create_harmonic(&ph, 200.0, 0.001, NULL, NULL, NULL));

    TFC_TEST_CHECK(tfBindTypes(&pc, &AType, &CType, 0));
    TFC_TEST_CHECK(tfBindTypes(&pc, &BType, &CType, 0));
    TFC_TEST_CHECK(tfBindTypes(&pa, &AType, &AType, 0));
    TFC_TEST_CHECK(tfBindTypes(&pab, &AType, &BType, 0));

    struct tfForceHandle forceBase;
    struct tfGaussianHandle force;
    TFC_TEST_CHECK(tfGaussian_init(&force, 5.0, 0.0, dt));
    TFC_TEST_CHECK(tfGaussian_toBase(&force, &forceBase));
    TFC_TEST_CHECK(tfBindForce(&forceBase, &AType));
    TFC_TEST_CHECK(tfBindForce(&forceBase, &BType));

    struct tfParticleListHandle plist;
    TFC_TEST_CHECK(tfParticleList_init(&plist));
    int numBParts;
    unsigned int pindex;
    struct tfParticleHandleHandle part;
    TFC_TEST_CHECK(tfParticleType_getNumParts(&BType, &numBParts));
    for(unsigned int i = 0; i < numBParts; i++) {
        TFC_TEST_CHECK(tfParticleType_getParticle(&BType, i, &part));
        TFC_TEST_CHECK(tfParticleList_insertP(&plist, &part, &pindex));
    }
    TFC_TEST_CHECK(tfBindBonds(&ph, &plist, 1.0, NULL, NULL, 0, NULL, NULL, NULL, NULL));

    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}