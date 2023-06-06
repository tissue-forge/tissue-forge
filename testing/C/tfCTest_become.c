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
    
    struct tfSimulatorConfigHandle config;
    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfInitC(&config, NULL, 0));

    struct tfStateSpeciesHandle S1, S2, S3;
    TFC_TEST_CHECK(tfStateSpecies_initS(&S1, "S1"));
    TFC_TEST_CHECK(tfStateSpecies_initS(&S2, "S2"));
    TFC_TEST_CHECK(tfStateSpecies_initS(&S3, "S3"));
    struct tfStateSpeciesListHandle slistA, slistB;
    TFC_TEST_CHECK(tfStateSpeciesList_init(&slistA));
    TFC_TEST_CHECK(tfStateSpeciesList_insert(&slistA, &S1));
    TFC_TEST_CHECK(tfStateSpeciesList_insert(&slistA, &S2));
    TFC_TEST_CHECK(tfStateSpeciesList_insert(&slistA, &S3));
    TFC_TEST_CHECK(tfStateSpeciesList_init(&slistB));
    TFC_TEST_CHECK(tfStateSpeciesList_insert(&slistB, &S1));
    TFC_TEST_CHECK(tfStateSpeciesList_insert(&slistB, &S2));
    TFC_TEST_CHECK(tfStateSpeciesList_insert(&slistB, &S3));

    struct tfParticleTypeHandle AType, BType;

    TFC_TEST_CHECK(tfParticleType_init(&AType));
    TFC_TEST_CHECK(tfParticleType_init(&BType));

    TFC_TEST_CHECK(tfParticleType_registerType(&AType));
    TFC_TEST_CHECK(tfParticleType_registerType(&BType));

    TFC_TEST_CHECK(tfParticleType_setName(&AType, "AType"));
    TFC_TEST_CHECK(tfParticleType_setRadius(&AType, 1.0));
    TFC_TEST_CHECK(tfParticleType_setSpecies(&AType, &slistA));

    TFC_TEST_CHECK(tfParticleType_setName(&BType, "BType"));
    TFC_TEST_CHECK(tfParticleType_setRadius(&BType, 4.0));
    TFC_TEST_CHECK(tfParticleType_setSpecies(&BType, &slistB));

    struct tfRenderingColorMapperHandle cmapA, cmapB;
    struct tfRenderingStyleHandle styleA, styleB;

    TFC_TEST_CHECK(tfRenderingColorMapper_init(&cmapA, "rainbow", 0.0, 1.0));
    TFC_TEST_CHECK(tfRenderingColorMapper_setMapParticleSpecies(&cmapA, &AType, "S2"));
    TFC_TEST_CHECK(tfRenderingStyle_init(&styleA));
    TFC_TEST_CHECK(tfRenderingStyle_setColorMapper(&styleA, &cmapA));
    TFC_TEST_CHECK(tfParticleType_setStyle(&AType, &styleA));
    
    TFC_TEST_CHECK(tfRenderingColorMapper_init(&cmapB, "rainbow", 0.0, 1.0));
    TFC_TEST_CHECK(tfRenderingColorMapper_setMapParticleSpecies(&cmapB, &BType, "S2"));
    TFC_TEST_CHECK(tfRenderingStyle_init(&styleB));
    TFC_TEST_CHECK(tfRenderingStyle_setColorMapper(&styleB, &cmapB));
    TFC_TEST_CHECK(tfParticleType_setStyle(&BType, &styleB));

    struct tfParticleHandleHandle part;
    int pid;
    TFC_TEST_CHECK(tfParticleType_createParticle(&AType, &pid, NULL, NULL));
    TFC_TEST_CHECK(tfParticleHandle_init(&part, pid));

    struct tfStateStateVectorHandle svec;
    struct tfStateSpeciesListHandle partSList;
    TFC_TEST_CHECK(tfParticleHandle_getSpecies(&part, &svec));
    TFC_TEST_CHECK(tfStateStateVector_getSpecies(&svec, &partSList));
    unsigned int sid;
    TFC_TEST_CHECK(tfStateSpeciesList_indexOf(&partSList, "S2", &sid));
    TFC_TEST_CHECK(tfStateStateVector_setItem(&svec, sid, 0.5));

    TFC_TEST_CHECK(tfParticleHandle_become(&part, &BType));

    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}