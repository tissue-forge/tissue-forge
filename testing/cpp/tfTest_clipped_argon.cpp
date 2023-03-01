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

#include "tfTest.h"
#include <TissueForge.h>


using namespace TissueForge;


// create a particle type
// all new Particle derived types are automatically
// registered with the universe

struct ArgonType : ParticleType {

    ArgonType() : ParticleType(true) {
        radius = 0.1;
        mass = 39.4;
        registerType();
    };
};


int main(int argc, char const *argv[])
{
    // new simulator
    Simulator::Config config;
    config.setWindowless(true);
    config.setWindowSize({900, 900});
    config.clipPlanes = {
        planeEquation({1, 1, 0.5}, {2, 2, 2}), 
        planeEquation({-1, 1, -1}, {5, 5, 5})
    };
    config.universeConfig.dim = {10, 10, 10};
    TF_TEST_CHECK(init(config));

    // create a potential representing a 12-6 Lennard-Jones potential
    // A The first parameter of the Lennard-Jones potential.
    // B The second parameter of the Lennard-Jones potential.
    // cutoff
    FloatP_t pot_tol = 0.001;
    Potential *pot = Potential::lennard_jones_12_6(0.275, 3.0, 9.5075e-06, 6.1545e-03, &pot_tol);

    ArgonType *Argon = new ArgonType();
    Argon = (ArgonType*)Argon->get();

    // bind the potential with the *TYPES* of the particles
    TF_TEST_CHECK(bind::types(pot, Argon, Argon));

    // uniform random cube
    Argon->factory(13000);

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 20));

    return S_OK;
}
