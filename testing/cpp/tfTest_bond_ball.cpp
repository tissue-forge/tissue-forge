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


struct AType : ParticleType {

    AType() : ParticleType(true) {
        radius = 0.5;
        mass = 5.0;
        dynamics = PARTICLE_OVERDAMPED;
        style->setColor("MediumSeaGreen");
        registerType();
    }

};

struct BType : ParticleType {

    BType() : ParticleType(true) {
        radius = 0.2;
        mass = 1.0;
        dynamics = PARTICLE_OVERDAMPED;
        style->setColor("skyblue");
        registerType();
    }

};

struct CType : ParticleType {

    CType() : ParticleType(true) {
        radius = 10.0;
        setFrozen(true);
        style->setColor("orange");
        registerType();
    }

};


HRESULT update(const event::TimeEvent &e) {
    BType B;
    ParticleType *B_p = B.get();
    std::cout << e.times_fired << ", " << B_p->items().getCenterOfMass() << std::endl;
    return S_OK;
};


int main(int argc, char const *argv[])
{
    FVector3 dim(30.);

    Simulator::Config config;
    config.setWindowless(true);
    config.universeConfig.dim = dim;
    config.universeConfig.cutoff = 5.0;
    config.universeConfig.dt = 0.0005;
    TF_TEST_CHECK(init(config));

    AType *A = new AType();
    BType *B = new BType();
    CType *C = new CType();
    A = (AType*)A->get();
    B = (BType*)B->get();
    C = (CType*)C->get();

    FVector3 c_pos = Universe::getCenter();
    (*C)(&c_pos);

    // make a ring of of 50 particles
    std::vector<FVector3> pts = points(PointsType::Ring, 100);
    for(auto &p : pts) {
        p = p * (B->radius + C->radius) + Universe::getCenter() - FVector3(0, 0, 1);
        std::cout << p << std::endl;
    }
    B->factory(pts.size(), &pts);

    FloatP_t pot_pc_m = 2.0;
    FloatP_t pot_pc_max = 5.0;
    Potential *pc = Potential::glj(30.0, &pot_pc_m, NULL, NULL, NULL, NULL, &pot_pc_max);
    
    FloatP_t pot_pa_m = 2.5;
    FloatP_t pot_pa_max = 3.0;
    Potential *pa = Potential::glj(3.0, &pot_pa_m, NULL, NULL, NULL, NULL, &pot_pa_max);
    
    FloatP_t pot_pb_m = 4.0;
    FloatP_t pot_pb_max = 1.0;
    Potential *pb = Potential::glj(1.0, &pot_pb_m, NULL, NULL, NULL, NULL, &pot_pb_max);
    
    FloatP_t pot_pab_m = 2.0;
    FloatP_t pot_pab_max = 1.0;
    Potential *pab = Potential::glj(1.0, &pot_pab_m, NULL, NULL, NULL, NULL, &pot_pab_max);
    
    Potential *ph = Potential::harmonic(200.0, 0.001);

    TF_TEST_CHECK(bind::types(pc, A, C));
    TF_TEST_CHECK(bind::types(pc, B, C));
    TF_TEST_CHECK(bind::types(pa, A, A));
    TF_TEST_CHECK(bind::types(pab, A, B));

    Gaussian *r = Force::random(5, 0);

    TF_TEST_CHECK(bind::force(r, A));
    TF_TEST_CHECK(bind::force(r, B));

    bind::bonds(ph, B->items(), 1.0);

    // Implement the callback
    event::TimeEventMethod update_ev(update);
    event::onTimeEvent(0.01, &update_ev);

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
