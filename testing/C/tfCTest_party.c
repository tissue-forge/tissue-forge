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


HRESULT vary_colors(struct tfEventTimeEventHandle *e) {
    tfFloatP_t utime;
    TFC_TEST_CHECK(tfUniverse_getTime(&utime));
    float rate = (float)2.0 * M_PI * utime;
    float sf = (sinf(rate) + 1.0) * 0.5;
    float sf2 = (sinf(2.0 * rate) + 1) * 0.5;
    float cf = (cosf(rate) + 1.0) * 0.5;

    float gridColor[] = {sf, 0.0, sf2};
    float sceneBoxColor[] = {sf2, sf, 0.0};
    float lightDirection[] = {3.0 * (2.0 * sf - 1.0), 3.0 * (2.0 * cf - 1.0), 2.0};
    float lightColor[] = {(sf + 1.0) * 0.5, (sf + 1.0) * 0.5, (sf + 1.0) * 0.5};
    float ambientColor[] = {sf, sf, sf};
    
    TFC_TEST_CHECK(tfSystem_setGridColor(gridColor));
    TFC_TEST_CHECK(tfSystem_setSceneBoxColor(sceneBoxColor));
    TFC_TEST_CHECK(tfSystem_setShininess(1000.0 * sf + 10.0));
    TFC_TEST_CHECK(tfSystem_setLightDirection(lightDirection));
    TFC_TEST_CHECK(tfSystem_setLightColor(lightColor));
    TFC_TEST_CHECK(tfSystem_setAmbientColor(ambientColor));
    
    return S_OK;
}

HRESULT passthrough(struct tfEventTimeEventHandle *e) {
    return S_OK;
}


int main(int argc, char** argv) {
    tfFloatP_t ATypeRadius = 0.1;

    struct tfSimulatorConfigHandle config;
    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfInitC(&config, NULL, 0));

    struct tfParticleTypeHandle AType;
    TFC_TEST_CHECK(tfParticleType_init(&AType));
    TFC_TEST_CHECK(tfParticleType_setRadius(&AType, ATypeRadius));
    TFC_TEST_CHECK(tfParticleType_registerType(&AType));

    struct tfPotentialHandle pot;
    TFC_TEST_CHECK(tfPotential_create_harmonic(&pot, 100.0, 0.3, NULL, NULL, NULL));
    TFC_TEST_CHECK(tfBindTypes(&pot, &AType, &AType, 0));

    tfFloatP_t *center = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
    TFC_TEST_CHECK(tfUniverse_getCenter(&center));

    tfFloatP_t disp[] = {ATypeRadius + 0.07, 0.0, 0.0};
    tfFloatP_t pos0[3], pos1[3];
    for(unsigned int i = 0; i < 3; i++) {
        pos0[i] = center[i] - disp[i];
        pos1[i] = center[i] + disp[i];
    }

    struct tfParticleHandleHandle p0, p1;
    int pid0, pid1;
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid0, pos0, NULL));
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid1, pos1, NULL));
    TFC_TEST_CHECK(tfParticleHandle_init(&p0, pid0));
    TFC_TEST_CHECK(tfParticleHandle_init(&p1, pid1));

    struct tfBondHandleHandle bh;
    TFC_TEST_CHECK(tfBondHandle_create(&bh, &pot, &p0, &p1));

    tfFloatP_t dt;
    TFC_TEST_CHECK(tfUniverse_getDt(&dt));

    struct tfEventTimeEventHandle e;
    tfEventTimeEventMethodHandleFcn invokeMethod = (tfEventTimeEventMethodHandleFcn)&vary_colors;
    tfEventTimeEventMethodHandleFcn predicateMethod = (tfEventTimeEventMethodHandleFcn)&passthrough;
    struct tfEventTimeEventTimeSetterEnumHandle timeSetterEnum;
    TFC_TEST_CHECK(tfEventTimeEventTimeSetterEnum_init(&timeSetterEnum));
    TFC_TEST_CHECK(tfEventOnTimeEvent(&e, dt, &invokeMethod, &predicateMethod, timeSetterEnum.DEFAULT, 0.0, -1.0));

    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}