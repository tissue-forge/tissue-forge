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


struct AType : ParticleType {

    AType() : ParticleType(true) {
        radius = 1.0;
        mass = 2.5;
        style->setColor("MediumSeaGreen");
        registerType();
    };

};

struct SphereType : ParticleType {

    SphereType() : ParticleType(true) {
        radius = 3.0;
        setFrozen(true);
        style->setColor("orange");
        registerType();
    };
    
};


int main(int argc, char const *argv[])
{
    // dimensions of universe
    FVector3 dim(30.f);

    FloatP_t dist = 3.9;
    FloatP_t offset = 6.0;

    BoundaryConditionsArgsContainer *bcArgs = new BoundaryConditionsArgsContainer();
    bcArgs->setValue("x", BOUNDARY_POTENTIAL);
    bcArgs->setValue("y", BOUNDARY_POTENTIAL);
    bcArgs->setValue("z", BOUNDARY_POTENTIAL);

    Simulator::Config config;
    config.setWindowless(true);
    config.universeConfig.dim = dim;
    config.universeConfig.cutoff = 7.0;
    config.universeConfig.spaceGridSize = {3, 3, 3};
    config.universeConfig.dt = 0.01;
    config.universeConfig.setBoundaryConditions(bcArgs);
    TF_TEST_CHECK(tfTest_init(config));

    AType *A = new AType();
    SphereType *Sphere = new SphereType();
    A = (AType*)A->get();
    Sphere = (SphereType*)Sphere->get();

    FloatP_t p_d = 100.0;
    FloatP_t p_a = 1.0;
    FloatP_t p_min = -3.0;
    FloatP_t p_max = 4.0;
    Potential *p = Potential::morse(&p_d, &p_a, NULL, &p_min, &p_max);

    TF_TEST_CHECK(bind::types(p, A, Sphere));

    BoundaryConditions *bcs = Universe::getBoundaryConditions();
    TF_TEST_CHECK(bind::boundaryCondition(p, &bcs->bottom, A));
    TF_TEST_CHECK(bind::boundaryCondition(p, &bcs->top, A));
    TF_TEST_CHECK(bind::boundaryCondition(p, &bcs->left, A));
    TF_TEST_CHECK(bind::boundaryCondition(p, &bcs->right, A));
    TF_TEST_CHECK(bind::boundaryCondition(p, &bcs->front, A));
    TF_TEST_CHECK(bind::boundaryCondition(p, &bcs->back, A));
    
    FVector3 pos = Universe::getCenter() + FVector3(5, 0, 0);
    (*Sphere)(&pos);

    // above the sphere
    pos = Universe::getCenter() + FVector3(5, 0, Sphere->radius + dist);
    (*A)(&pos);

    // bottom of simulation
    pos = {Universe::getCenter()[0], Universe::getCenter()[1], dist};
    (*A)(&pos);

    // top of simulation
    pos = {Universe::getCenter()[0], Universe::getCenter()[1], dim[2] - dist};
    (*A)(&pos);

    // left of simulation
    pos = {dist, Universe::getCenter()[1] - offset, Universe::getCenter()[2]};
    (*A)(&pos);

    // right of simulation
    pos = {dim[0] - dist, Universe::get()->getCenter()[1] + offset, Universe::get()->getCenter()[2]};
    (*A)(&pos);

    // front of simulation
    pos = {Universe::get()->getCenter()[0], dist, Universe::get()->getCenter()[2]};
    (*A)(&pos);

    // back of simulation
    pos = {Universe::get()->getCenter()[0], dim[1] - dist, Universe::get()->getCenter()[2]};
    (*A)(&pos);

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
