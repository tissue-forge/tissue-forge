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


void forceFunction(struct tfCustomForceHandle *fh, tfFloatP_t *f) {
    f[0] = 0.1;
    f[1] = 0.0;
    f[2] = 0.0;
}


int main(int argc, char** argv) {
    tfFloatP_t dt = 0.1;
    tfFloatP_t *dim = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    dim[0] = 15.0; dim[1] = 12.0; dim[2] = 10.0;
    int *cells = (int*)malloc(3 * sizeof(int));
    cells[0] = 7; cells[1] = 6; cells[2] = 5;
    tfFloatP_t cutoff = 0.5;

    struct tfBoundaryConditionsArgsContainerHandle bargs;
    tfFloatP_t *bvel = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    for(unsigned int i = 0; i < 3; i++) bvel[i] = 0.0;
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_init(&bargs));
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_setVelocity(&bargs, "top", bvel));
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_setVelocity(&bargs, "bottom", bvel));

    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;
    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfUniverseConfig_setDt(&uconfig, dt));
    TFC_TEST_CHECK(tfUniverseConfig_setDim(&uconfig, dim));
    TFC_TEST_CHECK(tfUniverseConfig_setCells(&uconfig, cells));
    TFC_TEST_CHECK(tfUniverseConfig_setCutoff(&uconfig, cutoff));
    TFC_TEST_CHECK(tfUniverseConfig_setBoundaryConditions(&uconfig, &bargs));
    TFC_TEST_CHECK(tfTest_initC(&config));

    struct tfParticleTypeStyleSpec ATypeStyleDef = tfParticleTypeStyleSpec_init();
    struct tfParticleTypeSpec ATypeDef = tfParticleTypeSpec_init();

    ATypeStyleDef.color = "seagreen";
    ATypeDef.radius = 0.05;
    ATypeDef.mass = 10.0;
    ATypeDef.style = &ATypeStyleDef;

    struct tfParticleTypeHandle AType;
    TFC_TEST_CHECK(tfParticleType_initD(&AType, ATypeDef));
    TFC_TEST_CHECK(tfParticleType_registerType(&AType));

    struct tfPotentialHandle dpd;
    tfFloatP_t dpd_alpha = 10.0;
    tfFloatP_t dpd_sigma = 1.0;
    TFC_TEST_CHECK(tfPotential_create_dpd(&dpd, &dpd_alpha, NULL, &dpd_sigma, NULL, NULL));
    TFC_TEST_CHECK(tfBindTypes(&dpd, &AType, &AType, 0));

    struct tfCustomForceHandle pressure;
    struct tfForceHandle pressure_base;
    struct tfUserForceFuncTypeHandle forceFunc;
    tfUserForceFuncTypeHandleFcn _forceFunction = (tfUserForceFuncTypeHandleFcn)forceFunction;
    TFC_TEST_CHECK(tfForce_EvalFcn_init(&forceFunc, &_forceFunction));
    TFC_TEST_CHECK(tfCustomForce_init(&pressure, &forceFunc, 0.0));

    TFC_TEST_CHECK(tfCustomForce_toBase(&pressure, &pressure_base));
    TFC_TEST_CHECK(tfBindForce(&pressure_base, &AType));

    int pid;
    for(unsigned int i = 0; i < 5000; i++) 
        TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, NULL, NULL));

    TFC_TEST_CHECK(tfTest_runQuiet(20));

    return 0;
}