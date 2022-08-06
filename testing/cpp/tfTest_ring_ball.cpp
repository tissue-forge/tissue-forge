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

#include "tfTest.h"
#include <TissueForge.h>


using namespace TissueForge;


struct BeadType : ParticleType {

    BeadType() : ParticleType(true) {
        mass = 1.0;
        radius = 0.1;
        dynamics = PARTICLE_OVERDAMPED;
        registerType();
    };

};


int main(int argc, char const *argv[])
{
    Simulator::Config config;
    config.universeConfig.dim = {20, 20, 20};
    config.universeConfig.cutoff = 8.0;
    config.setWindowless(true);
    TF_TEST_CHECK(init(config));

    BeadType *Bead = new BeadType();
    Bead = (BeadType*)Bead->get();

    // simple harmonic potential to pull particles
    FloatP_t pot_max = 3.0;
    Potential *pot = Potential::harmonic(1.0, 0.1, NULL, &pot_max);

    // make a ring of of 50 particles
    std::vector<FVector3> pts = points(PointsType::Ring, 50);

    // constuct a particle for each position, make
    // a list of particles
    ParticleList beads;
    for(auto &p : pts) {
        FVector3 pos = p * 5 + Universe::getCenter();
        beads.insert((*Bead)(&pos));
    }

    // create an explicit bond for each pair in the
    // list of particles. The bind_pairwise method
    // searches for all possible pairs within a cutoff
    // distance and connects them with a bond.
    TF_TEST_CHECK(bind::bonds(pot, &beads, 1));

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
