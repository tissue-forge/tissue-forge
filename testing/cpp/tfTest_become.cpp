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


static std::vector<std::string> speciesNames = {"S1", "S2", "S3"};


struct AType : ParticleType {

    AType() : ParticleType(true) {
        radius = 1.0;
        species = new state::SpeciesList();
        for(auto &s : speciesNames) 
            species->insert(s);
        style = new rendering::Style();
        style->newColorMapper(this, "S2");

        registerType();
    };

};

struct BType : ParticleType {

    BType() : ParticleType(true) {
        radius = 4.0;
        species = new state::SpeciesList();
        for(auto &s : speciesNames) 
            species->insert(s);
        style = new rendering::Style();
        style->newColorMapper(this, "S2");

        registerType();
    };

};


int main(int argc, char const *argv[])
{
    Simulator::Config config;
    config.setWindowless(true);
    TF_TEST_CHECK(init(config));

    AType *A = new AType();
    A = (AType*)A->get();
    BType *B = new BType();
    B = (BType*)B->get();

    ParticleHandle *o = (*A)();
    state::StateVector *ostate = o->getSpecies();
    ostate->setItem(ostate->species->index_of("S2"), 0.5);
    TF_TEST_CHECK(o->become(B));

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
