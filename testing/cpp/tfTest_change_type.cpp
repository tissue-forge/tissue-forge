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


using namespace TissueForge;


struct AType : ParticleType {

    AType() : ParticleType(true) {
        radius = 0.1;
        dynamics = PARTICLE_OVERDAMPED;
        style = new rendering::Style("MediumSeaGreen");
        registerType();
    };

};

struct BType : ParticleType {

    BType() : ParticleType(true) {
        radius = 0.1;
        dynamics = PARTICLE_OVERDAMPED;
        style = new rendering::Style("skyblue");
        registerType();
    };
    
};


int main(int argc, char const *argv[])
{
    Simulator::Config config;
    config.setWindowless(true);
    config.universeConfig.cutoff = 3.0;
    TF_TEST_CHECK(tfTest_init(config));

    AType *A = new AType();
    BType *B = new BType();
    A = (AType*)A->get();
    B = (BType*)B->get();

    FloatP_t pot_min = 0.01;
    FloatP_t pot_max = 3.0;
    Potential *p = Potential::coulomb(0.5, &pot_min, &pot_max);
    Potential *q = Potential::coulomb(0.5, &pot_min, &pot_max);
    Potential *r = Potential::coulomb(2.0, &pot_min, &pot_max);

    TF_TEST_CHECK(bind::types(p, A, A));
    TF_TEST_CHECK(bind::types(q, B, B));
    TF_TEST_CHECK(bind::types(r, A, B));

    int nr_parts = 1000;
    std::vector<FVector3> pos;
    pos.reserve(nr_parts);
    for(auto &p_pos : randomPoints(PointsType::SolidCube, 1000)) 
        pos.push_back(p_pos * 10 + Universe::getCenter());

    A->factory(nr_parts, &pos);

    ParticleHandle *a = A->items().item(0);
    ParticleList n_list = a->neighbors(5.0);
    for(int i = 0; i < n_list.nr_parts; i++) 
        n_list.item(i)->become(B);

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
