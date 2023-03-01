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


struct BType : ParticleType {

    BType() : ParticleType(true) {
        radius = 0.25;
        dynamics = PARTICLE_OVERDAMPED;
        mass = 15.0;
        style->setColor("skyblue");
        registerType();
    };

};

struct CType : ClusterParticleType {

    CType() : ClusterParticleType(true) {
        radius = 5.0;
        registerType();
    };

};

struct YolkType : ParticleType {

    YolkType() : ParticleType(true) {
        radius = 10.0;
        mass = 1000000.0;
        dynamics = PARTICLE_OVERDAMPED;
        setFrozen(true);
        style->setColor("gold");
        registerType();
    };

};


static YolkType *Yolk = NULL;
static int yolk_id = -1;


static HRESULT split(const event::ParticleTimeEvent &event) {
    ClusterParticleHandle *particle = (ClusterParticleHandle*)event.targetParticle;
    ClusterParticleType *ptype = (ClusterParticleType*)event.targetType;

    ParticleHandle yolk(yolk_id);
    FVector3 axis = particle->getPosition() - yolk.getPosition();
    particle->split(&axis);
    return S_OK;
}


int main(int argc, char const *argv[])
{
    FVector3 pos;

    Simulator::Config config;
    config.setWindowless(true);
    config.universeConfig.dim = {30, 30, 30};
    config.universeConfig.cutoff = 3;
    config.universeConfig.dt = 0.001;
    config.universeConfig.spaceGridSize = {6, 6, 6};
    TF_TEST_CHECK(init(config));

    BType *B = new BType();
    CType *C = new CType();
    Yolk = new YolkType();
    B = (BType*)B->get();
    C = (CType*)C->get();
    Yolk = (YolkType*)Yolk->get();

    FloatP_t total_height = 2.0 * Yolk->radius + 2.0 * C->radius;
    FloatP_t yshift = 1.5 * (total_height / 2.0 - Yolk->radius);
    FloatP_t cshift = total_height / 2.0 - C->radius - 1.0;

    pos = Universe::getCenter() - FVector3(0, 0, -yshift);
    ParticleHandle *yolk = (*Yolk)(&pos);
    yolk_id = yolk->id;

    pos = yolk->getPosition() + FVector3(0, 0, yolk->getRadius() + C->radius - 10.0);
    ParticleHandle *c = (*C)(&pos);

    B->factory(8000);

    FloatP_t pot_a = 6;

    FloatP_t pb_d = 1.0;
    FloatP_t pb_r0 = 0.5;
    FloatP_t pb_min = 0.01;
    FloatP_t pb_max = 3.0;
    bool pb_shifted = false;
    Potential *pb = Potential::morse(&pb_d, &pot_a, &pb_r0, &pb_min, &pb_max, NULL, &pb_shifted);

    FloatP_t pub_d = 1.0;
    FloatP_t pub_r0 = 0.5;
    FloatP_t pub_min = 0.01;
    FloatP_t pub_max = 3.0;
    bool pub_shifted = false;
    Potential *pub = Potential::morse(&pub_d, &pot_a, &pub_r0, &pub_min, &pub_max, NULL, &pub_shifted);

    FloatP_t py_d = 0.1;
    FloatP_t py_min = -5.0;
    FloatP_t py_max = 1.0;
    Potential *py = Potential::morse(&py_d, &pot_a, NULL, &py_min, &py_max);

    Gaussian *rforce = Force::random(500.0, 0.0, 0.0001);

    TF_TEST_CHECK(bind::force(rforce, B));
    TF_TEST_CHECK(bind::types(pb, B, B, true));
    TF_TEST_CHECK(bind::types(pub, B, B));
    TF_TEST_CHECK(bind::types(py, Yolk, B));

    event::ParticleTimeEventMethod invokeMethod(split);
    event::onParticleTimeEvent(C, 1.0, &invokeMethod, NULL, (unsigned int)event::ParticleTimeEventParticleSelectorEnum::LARGEST);

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
