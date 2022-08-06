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


// create a particle type

struct ArgonType : ParticleType {

    ArgonType() : ParticleType(true) {
        mass = 39.4;
        target_energy = 10000.0;
        registerType();
    }

};


int main(int argc, char const *argv[])
{
    // potential cutoff distance
    FloatP_t cutoff = 1.0;

    // dimensions of universe
    FVector3 dim(10.);

    // new simulator
    Simulator::Config config;
    config.setWindowless(true);
    config.universeConfig.dim = dim;
    config.universeConfig.cutoff = cutoff;
    TF_TEST_CHECK(init(config));

    // create a potential representing a 12-6 Lennard-Jones potential
    FloatP_t pot_tol = 0.001;
    Potential *pot = Potential::lennard_jones_12_6(0.275, cutoff, 9.5075e-06, 6.1545e-03, &pot_tol);

    // Register and get the particle type; registration always only occurs once
    ArgonType *Argon = new ArgonType();
    Argon = (ArgonType*)Argon->get();

    // bind the potential with the *TYPES* of the particles
    TF_TEST_CHECK(bind::types(pot, Argon, Argon));

    // create a thermostat, coupling time constant determines how rapidly the
    // thermostat operates, smaller numbers mean thermostat acts more rapidly
    Berendsen *tstat = Force::berendsen_tstat(10.0);

    // bind it just like any other force
    TF_TEST_CHECK(bind::force(tstat, Argon));

    int nr_parts = 100;
    std::vector<FVector3> velocities;
    velocities.reserve(nr_parts);
    for(int i = 0; i < nr_parts; i++) 
        velocities.push_back(randomUnitVector() * 0.1);
    Argon->factory(nr_parts, NULL, &velocities);

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
