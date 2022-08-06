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

#include <TissueForge.h>

using namespace TissueForge;


// create a particle type

struct ArgonType : ParticleType {

    ArgonType() : ParticleType(true) {
        radius = 0.1;
        mass = 39.4;
        registerType();
    }

};


int main(int argc, char const *argv[])
{
    // dimensions of universe
    FVector3 dim(10.);

    // new simulator
    Simulator::Config config;
    config.universeConfig.dim = dim;
    config.universeConfig.spaceGridSize = {5, 5, 5};
    config.universeConfig.cutoff = 1.;
    init(config);

    // create a potential representing a 12-6 Lennard-Jones potential
    FloatP_t pot_tol = 0.001;
    Potential *pot = Potential::lennard_jones_12_6(0.275, 1.0, 9.5075e-06, 6.1545e-03, &pot_tol);

    // Register and get the particle type; registration always only occurs once
    ArgonType *Argon = new ArgonType();
    Argon = (ArgonType*)Argon->get();

    // bind the potential with the *TYPES* of the particles
    bind::types(pot, Argon, Argon);

    // random cube
    Argon->factory(2500);

    // run the simulator
    show();

    return S_OK;
}
