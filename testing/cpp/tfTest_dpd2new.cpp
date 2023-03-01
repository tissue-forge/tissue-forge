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
        radius = 0.3;
        dynamics = PARTICLE_OVERDAMPED;
        mass = 10.0;
        style->setColor("seagreen");
        registerType();
    };

};


int main(int argc, char const *argv[])
{
    BoundaryConditionsArgsContainer bcArgs;
    bcArgs.setValue("x", BOUNDARY_PERIODIC);
    bcArgs.setValue("y", BOUNDARY_PERIODIC);
    bcArgs.setValue("z", BOUNDARY_NO_SLIP);

    Simulator::Config config;
    config.setWindowless(true);
    config.universeConfig.dt = 0.1;
    config.universeConfig.dim = {15, 12, 10};
    TF_TEST_CHECK(init(config));

    AType *A = new AType();
    A = (AType*)A->get();

    FloatP_t dpd_alpha = 0.3;
    FloatP_t dpd_gamma = 1.0;
    FloatP_t dpd_sigma = 1.0;
    FloatP_t dpd_cutoff = 0.6;
    Potential *dpd = Potential::dpd(&dpd_alpha, &dpd_gamma, &dpd_sigma, &dpd_cutoff);

    FloatP_t dpd_wall_alpha = 0.5;
    FloatP_t dpd_wall_gamma = 10.0;
    FloatP_t dpd_wall_sigma = 1.0;
    FloatP_t dpd_wall_cutoff = 0.1;
    Potential *dpd_wall = Potential::dpd(&dpd_wall_alpha, &dpd_wall_gamma, &dpd_wall_sigma, &dpd_wall_cutoff);

    FloatP_t dpd_left_alpha = 1.0;
    FloatP_t dpd_left_gamma = 100.0;
    FloatP_t dpd_left_sigma = 0.0;
    FloatP_t dpd_left_cutoff = 0.5;
    Potential *dpd_left = Potential::dpd(&dpd_left_alpha, &dpd_left_gamma, &dpd_left_sigma, &dpd_left_cutoff);

    TF_TEST_CHECK(bind::types(dpd, A, A));
    TF_TEST_CHECK(bind::boundaryCondition(dpd_wall, &Universe::getBoundaryConditions()->top, A));
    TF_TEST_CHECK(bind::boundaryCondition(dpd_left, &Universe::getBoundaryConditions()->left, A));

    A->factory(1000);

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
