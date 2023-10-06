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


tfFloatP_t lam = -0.5;
tfFloatP_t mu = 1.0;
unsigned int s = 3;


tfFloatP_t He(tfFloatP_t r, unsigned int n) {
    if(n == 0) {
        return 1.0;
    }
    else if(n == 1) {
        return r;
    } 
    else {
        return r * He(r, n - 1) - (n - 1) * He(r, n - 2);
    }
}

unsigned int factorial(unsigned int k) {
    unsigned int result = 1;
    for(unsigned int j = 1; j <= k; j++) 
        result *= j;
    return result;
}

tfFloatP_t dgdr(tfFloatP_t r, unsigned int n) {
    tfFloatP_t result = 0.0;
    for(unsigned int k = 0; k < s; k++) {
        if(2 * k >= n) {
            result += (tfFloatP_t)factorial(2 * k) / (tfFloatP_t)factorial(2 * k - n) * (lam + (tfFloatP_t)k) * pow(mu, (tfFloatP_t)k) / (tfFloatP_t)factorial(k) * pow(r, 2.0 * (tfFloatP_t)k);
        }
    }
    return result / pow(r, (tfFloatP_t)n);
}

tfFloatP_t u_n(tfFloatP_t r, unsigned int n) {
    return pow(-1, (tfFloatP_t)n) * He(r, n) * lam * exp(-mu * pow(r, 2.0));
}

tfFloatP_t f_n(tfFloatP_t r, unsigned int n) {
    tfFloatP_t w_n = 0.0;
    for(unsigned int j = 0; j <= n; j++) {
        w_n += (tfFloatP_t)factorial(n) / (tfFloatP_t)factorial(j) / (tfFloatP_t)factorial(n - j) * dgdr(r, j) * u_n(r, n - j);
    }
    return 10.0 * (u_n(r, n) + w_n / lam);
}

tfFloatP_t f(tfFloatP_t r) {
    return f_n(r, 0);
}


int main(int argc, char** argv) {
    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;
    struct tfBoundaryConditionsArgsContainerHandle bargs;

    tfFloatP_t bcvel[] = {0.0, 0.0, 0.0};
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_init(&bargs));
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_setVelocity(&bargs, "left", bcvel));
    TFC_TEST_CHECK(tfBoundaryConditionsArgsContainer_setVelocity(&bargs, "right", bcvel));

    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfUniverseConfig_setCutoff(&uconfig, 5.0));
    TFC_TEST_CHECK(tfUniverseConfig_setBoundaryConditions(&uconfig, &bargs));
    TFC_TEST_CHECK(tfTest_initC(&config));

    struct tfParticleTypeStyleSpec WellStyleDef = tfParticleTypeStyleSpec_init();
    struct tfParticleTypeSpec WellTypeDef = tfParticleTypeSpec_init();

    WellStyleDef.visible = 0;
    WellTypeDef.frozen = 1;
    WellTypeDef.style = &WellStyleDef;

    struct tfParticleTypeHandle WellType, SmallType;
    TFC_TEST_CHECK(tfParticleType_initD(&WellType, WellTypeDef));
    TFC_TEST_CHECK(tfParticleType_registerType(&WellType));
    TFC_TEST_CHECK(tfParticleType_init(&SmallType));
    TFC_TEST_CHECK(tfParticleType_setRadius(&SmallType, 0.1));
    TFC_TEST_CHECK(tfParticleType_setFrozenY(&SmallType, 1));
    TFC_TEST_CHECK(tfParticleType_registerType(&SmallType));

    struct tfPotentialHandle pot_c;
    TFC_TEST_CHECK(tfPotential_create_custom(&pot_c, 0.0, 5.0, f, NULL, NULL, NULL, NULL));
    TFC_TEST_CHECK(tfBindTypes(&pot_c, &WellType, &SmallType, 0));

    tfFloatP_t *center = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    TFC_TEST_CHECK(tfUniverse_getCenter(&center));
    tfFloatP_t *dim = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    TFC_TEST_CHECK(tfUniverse_getDim(&dim));

    int pid;
    TFC_TEST_CHECK(tfParticleType_createParticle(&WellType, &pid, center, NULL));

    tfFloatP_t position[] = {0.0, center[1], center[2]};
    for(unsigned int i = 0; i < 20; i++) {
        position[0] = (tfFloatP_t)(i + 1) / 21 * dim[0];
        TFC_TEST_CHECK(tfParticleType_createParticle(&SmallType, &pid, position, NULL));
    }

    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}