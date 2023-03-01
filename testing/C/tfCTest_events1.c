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


HRESULT split(struct tfEventParticleTimeEventHandle *e) {
    struct tfParticleTypeHandle ptype;
    TFC_TEST_CHECK(tfEventParticleTimeEvent_getTargetType(e, &ptype));
    struct tfParticleHandleHandle targetParticle;
    if(tfEventParticleTimeEvent_getTargetParticle(e, &targetParticle) != S_OK) {
        tfFloatP_t *center = (tfFloatP_t*)malloc(3 * sizeof(tfFloatP_t));
        TFC_TEST_CHECK(tfUniverse_getCenter(&center));
        int pid;
        TFC_TEST_CHECK(tfParticleType_createParticle(&ptype, &pid, center, NULL));
    }
    else {
        tfFloatP_t pradius;
        TFC_TEST_CHECK(tfParticleType_getRadius(&ptype, &pradius));
        struct tfParticleHandleHandle p;
        TFC_TEST_CHECK(tfParticleHandle_split(&targetParticle, &p));
        TFC_TEST_CHECK(tfParticleHandle_setRadius(&targetParticle, pradius));
        TFC_TEST_CHECK(tfParticleHandle_setRadius(&p, pradius));
    }
    return S_OK;
}

HRESULT passthrough(struct tfEventParticleTimeEventHandle *e) {
    return S_OK;
}


int main(int argc, char** argv) {
    struct tfSimulatorConfigHandle config;
    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfInitC(&config, NULL, 0));

    struct tfParticleTypeHandle MyCellType;
    TFC_TEST_CHECK(tfParticleType_init(&MyCellType));
    TFC_TEST_CHECK(tfParticleType_setMass(&MyCellType, 39.4));
    TFC_TEST_CHECK(tfParticleType_setRadius(&MyCellType, 0.2));
    TFC_TEST_CHECK(tfParticleType_setTargetTemperature(&MyCellType, 50.0));
    TFC_TEST_CHECK(tfParticleType_registerType(&MyCellType));

    struct tfPotentialHandle pot;
    tfFloatP_t pot_tol = 1.0e-3;
    TFC_TEST_CHECK(tfPotential_create_lennard_jones_12_6(&pot, 0.275, 1.0, 9.5075e-06, 6.1545e-03, &pot_tol));
    TFC_TEST_CHECK(tfBindTypes(&pot, &MyCellType, &MyCellType, 0));

    struct tfBerendsenHandle force;
    struct tfForceHandle force_base;
    TFC_TEST_CHECK(tfBerendsen_init(&force, 10.0));
    TFC_TEST_CHECK(tfBerendsen_toBase(&force, &force_base));
    TFC_TEST_CHECK(tfBindForce(&force_base, &MyCellType));
    
    struct tfEventParticleTimeEventTimeSetterEnumHandle setterEnum;
    TFC_TEST_CHECK(tfEventParticleTimeEventTimeSetterEnum_init(&setterEnum));

    struct tfEventParticleTimeEventParticleSelectorEnumHandle selectorEnum;
    TFC_TEST_CHECK(tfEventParticleTimeEventParticleSelectorEnum_init(&selectorEnum));

    tfEventParticleTimeEventMethodHandleFcn invokeMethod = (tfEventParticleTimeEventMethodHandleFcn)&split;
    tfEventParticleTimeEventMethodHandleFcn predicateMethod = (tfEventParticleTimeEventMethodHandleFcn)&passthrough;

    struct tfEventParticleTimeEventHandle pevent;
    TFC_TEST_CHECK(tfEventOnParticleTimeEvent(
        &pevent, &MyCellType, 0.05, &invokeMethod, &predicateMethod, setterEnum.EXPONENTIAL, 0.0, -1.0, selectorEnum.DEFAULT
    ));

    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}