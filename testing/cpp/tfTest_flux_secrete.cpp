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


static std::vector<std::string> speciesNames = {"S1", "S2", "S3"};


struct AType : ParticleType {

    AType() : ParticleType(true) {
        radius = 0.1;
        species = new state::SpeciesList();
        for(auto &s : speciesNames) species->insert(s);
        style->mapper = new rendering::ColorMapper();
        style->mapper->setMapParticleSpecies(this, "S1");
        registerType();
    };

};

struct BType : ParticleType {

    BType() : ParticleType(true) {
        radius = 0.1;
        species = new state::SpeciesList();
        for(auto &s : speciesNames) species->insert(s);
        style->mapper = new rendering::ColorMapper();
        style->mapper->setMapParticleSpecies(this, "S1");
        registerType();
    };
    
};


HRESULT spew(const event::ParticleTimeEvent &event) {
    std::cout << "Spew" << std::endl;
    state::StateVector *sv = event.targetParticle->getSpecies();
    int32_t s_idx = sv->species->index_of("S1");
    sv->setItem(s_idx, 500);
    state::SpeciesValue(sv, s_idx).secrete(250.0, 1.0);
    return S_OK;
}


int main(int argc, char const *argv[])
{
    BoundaryConditionsArgsContainer *bcArgs = new BoundaryConditionsArgsContainer();
    bcArgs->setValueAll(BOUNDARY_FREESLIP);

    Simulator::Config config;
    config.universeConfig.dim = {6.5, 6.5, 6.5};
    config.universeConfig.setBoundaryConditions(bcArgs);
    config.setWindowless(true);
    TF_TEST_CHECK(tfTest_init(config));

    AType *A = new AType();
    BType *B = new BType();
    A = (AType*)A->get();
    B = (BType*)B->get();

    Fluxes::flux(A, A, "S1", 5, 0.005);

    A->factory(10000);

    // Grab a particle
    ParticleHandle *o = A->parts.item(0);

    // Change type to B, since there is no flux rule between A and B
    o->become(B);

    event::ParticleTimeEventMethod spew_e(spew);
    event::onParticleTimeEvent(B, 0.3, &spew_e);

    // run the simulator
    TF_TEST_CHECK(step(0.35));

    return S_OK;
}
