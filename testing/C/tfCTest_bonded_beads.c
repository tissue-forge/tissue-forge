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
    tfFloatP_t cutoff = 8.0;
    tfFloatP_t dim[] = {20.0, 20.0, 20.0};

    struct tfSimulatorConfigHandle config;
    struct tfUniverseConfigHandle uconfig;

    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfSimulatorConfig_getUniverseConfig(&config, &uconfig));
    TFC_TEST_CHECK(tfUniverseConfig_setDim(&uconfig, dim));
    TFC_TEST_CHECK(tfUniverseConfig_setCutoff(&uconfig, cutoff));

    TFC_TEST_CHECK(tfTest_initC(&config));

    tfFloatP_t dt;
    TFC_TEST_CHECK(tfUniverse_getDt(&dt));

    struct tfParticleDynamicsEnumHandle dynEnums;
    TFC_TEST_CHECK(tfParticleDynamics_init(&dynEnums));

    struct tfParticleTypeHandle BeadType;
    TFC_TEST_CHECK(tfParticleType_init(&BeadType));
    TFC_TEST_CHECK(tfParticleType_setMass(&BeadType, 0.4));
    TFC_TEST_CHECK(tfParticleType_setRadius(&BeadType, 0.2));
    TFC_TEST_CHECK(tfParticleType_setDynamics(&BeadType, dynEnums.PARTICLE_OVERDAMPED));
    TFC_TEST_CHECK(tfParticleType_registerType(&BeadType));

    struct tfPotentialHandle pot_bb, pot_bond, pot_ang;

    tfFloatP_t bb_min = 0.1, bb_max = 1.0;
    TFC_TEST_CHECK(tfPotential_create_coulomb(&pot_bb, 0.1, &bb_min, &bb_max, NULL, NULL));
    TFC_TEST_CHECK(tfBindTypes(&pot_bb, &BeadType, &BeadType, 0));

    tfFloatP_t bond_min=0.0, bond_max = 2.0;
    TFC_TEST_CHECK(tfPotential_create_harmonic(&pot_bond, 0.4, 0.2, &bond_min, &bond_max, NULL));

    tfFloatP_t ang_tol = 0.01;
    TFC_TEST_CHECK(tfPotential_create_harmonic_angle(&pot_ang, 0.2, 0.85 * M_PI, NULL, NULL, &ang_tol));

    struct tfGaussianHandle force_rnd;
    struct tfForceHandle force_rnd_base;
    TFC_TEST_CHECK(tfGaussian_init(&force_rnd, 0.1, 0.0, dt));
    TFC_TEST_CHECK(tfGaussian_toBase(&force_rnd, &force_rnd_base));
    TFC_TEST_CHECK(tfBindForce(&force_rnd_base, &BeadType));

    unsigned int numBeads = 80;
    tfFloatP_t xx[numBeads];
    xx[0] = 4.0;
    for(unsigned int i = 1; i < numBeads; i++) {
        xx[i] = xx[i - 1] + 0.15;
    }

    tfFloatP_t pos0[] = {xx[0], 10.0, 10.0};
    struct tfParticleHandleHandle bead, p, n;
    int beadid, pid, nid;
    struct tfAngleHandleHandle angle;
    TFC_TEST_CHECK(tfParticleType_createParticle(&BeadType, &beadid, pos0, NULL));
    TFC_TEST_CHECK(tfParticleHandle_init(&bead, beadid));
    for(unsigned int i = 1; i < numBeads; i++) {
        pos0[0] = xx[i];
        TFC_TEST_CHECK(tfParticleType_createParticle(&BeadType, &nid, pos0, NULL));
        TFC_TEST_CHECK(tfParticleHandle_init(&n, nid));
        if(i > 1) {
            TFC_TEST_CHECK(tfAngleHandle_create(&angle, &pot_ang, &p, &bead, &n));
        }
        p = bead;
        bead = n;
    }

    TFC_TEST_CHECK(tfTest_runQuiet(100));

    return 0;
}