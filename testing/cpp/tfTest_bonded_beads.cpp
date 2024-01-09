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

#include "tfTest.h"


using namespace TissueForge;


struct BeadType : ParticleType {

    BeadType() : ParticleType(true) {
        mass = 0.4;
        radius = 0.2;
        dynamics = PARTICLE_OVERDAMPED;
        registerType();
    };

};


int main(int argc, char const *argv[])
{
    // new simulator
    Simulator::Config config;
    config.universeConfig.dim = {20., 20., 20.};
    config.universeConfig.cutoff = 8.0;
    config.setWindowless(true);
    TF_TEST_CHECK(tfTest_init(config));

    BeadType *Bead = new BeadType();
    Bead = (BeadType*)Bead->get();

    FloatP_t pot_bb_min = 0.1;
    FloatP_t pot_bb_max = 1.0;
    Potential *pot_bb = Potential::coulomb(0.1, &pot_bb_min, &pot_bb_max);

    // hamonic bond between particles
    FloatP_t pot_bond_max = 2.0;
    Potential *pot_bond = Potential::harmonic(0.4, 0.2, NULL, &pot_bond_max);

    // angle bond potential
    FloatP_t pot_ang_tol = 0.01;
    Potential *pot_ang = Potential::harmonic_angle(0.01, 0.85 * M_PI, NULL, NULL, &pot_ang_tol);

    // bind the potential with the *TYPES* of the particles
    TF_TEST_CHECK(bind::types(pot_bb, Bead, Bead));

    // create a random force. In overdamped dynamcis, we neeed a random force to
    // enable the objects to move around, otherwise they tend to get trapped
    // in a potential
    Gaussian *rforce = Force::random(0.1, 0);

    // bind it just like any other force
    TF_TEST_CHECK(bind::force(rforce, Bead));

    // Place particles
    ParticleHandle *p = NULL;     // previous bead
    ParticleHandle *bead;         // current bead
    ParticleHandle *n;            // new bead

    FVector3 pos(4.0, 10.f, 10.f);
    bead = (*Bead)(&pos);

    while(pos[0] < 16.f) {
        pos[0] += 0.15;
        n = (*Bead)(&pos);
        Bond::create(pot_bond, bead, n);
        if(p != NULL) 
            Angle::create(pot_ang, p, bead, n);
        p = bead;
        bead = n;
    }

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
