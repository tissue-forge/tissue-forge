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
    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;

    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfUniverseConfig_setCutoff(&uconfig, 3.0));
    TFC_TEST_CHECK(tfTest_initC(&config));

    struct tfParticleTypeStyleSpec AStyleDef = tfParticleTypeStyleSpec_init();
    struct tfParticleTypeStyleSpec BStyleDef = tfParticleTypeStyleSpec_init();
    struct tfParticleTypeSpec ATypeDef = tfParticleTypeSpec_init();
    struct tfParticleTypeSpec BTypeDef = tfParticleTypeSpec_init();

    AStyleDef.color = "MediumSeaGreen";
    ATypeDef.radius = 0.1;
    ATypeDef.dynamics = 1;
    ATypeDef.style = &AStyleDef;

    BStyleDef.color = "skyblue";
    BTypeDef.radius = 0.1;
    BTypeDef.dynamics = 1;
    BTypeDef.style = &BStyleDef;

    struct tfParticleTypeHandle AType, BType;
    TFC_TEST_CHECK(tfParticleType_initD(&AType, ATypeDef));
    TFC_TEST_CHECK(tfParticleType_initD(&BType, BTypeDef));

    TFC_TEST_CHECK(tfParticleType_registerType(&AType));
    TFC_TEST_CHECK(tfParticleType_registerType(&BType));

    struct tfPotentialHandle p, q, r;
    tfFloatP_t pot_min = 0.01;
    tfFloatP_t pot_max = 3.0;
    TFC_TEST_CHECK(tfPotential_create_coulomb(&p, 0.5, &pot_min, &pot_max, NULL, NULL));
    TFC_TEST_CHECK(tfPotential_create_coulomb(&q, 0.5, &pot_min, &pot_max, NULL, NULL));
    TFC_TEST_CHECK(tfPotential_create_coulomb(&r, 2.0, &pot_min, &pot_max, NULL, NULL));

    TFC_TEST_CHECK(tfBindTypes(&p, &AType, &AType, 0));
    TFC_TEST_CHECK(tfBindTypes(&q, &BType, &BType, 0));
    TFC_TEST_CHECK(tfBindTypes(&r, &AType, &BType, 0));

    struct tfPointsTypeHandle ptTypes;
    TFC_TEST_CHECK(tfPointsType_init(&ptTypes));

    tfFloatP_t *center = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    TFC_TEST_CHECK(tfUniverse_getCenter(&center));

    tfFloatP_t *pos;
    int pid;
    unsigned int numPos = 1000;
    TFC_TEST_CHECK(tfRandomPoints(ptTypes.SolidCube, numPos, 0, 0, 0, &pos));

    for(unsigned int i = 0; i < numPos; i++) {
        tfFloatP_t partPos[3];
        for(unsigned int j = 0; j < 3; j++) 
            partPos[j] = pos[3 * i + j] * 10 + center[j];
        TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, partPos, NULL));
    }

    struct tfParticleHandleHandle *neighbors;
    int numNeighbors;
    struct tfParticleHandleHandle part;
    TFC_TEST_CHECK(tfParticleType_getParticle(&AType, 0, &part));
    TFC_TEST_CHECK(tfParticleHandle_neighborsD(&part, 5.0, &neighbors, &numNeighbors));
    for(unsigned int i = 0; i < numNeighbors; i++) {
        TFC_TEST_CHECK(tfParticleHandle_become(&neighbors[i], &BType));
    }

    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}