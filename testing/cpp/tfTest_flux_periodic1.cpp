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


struct AType : ParticleType {

    AType() : ParticleType(true) {
        species = new state::SpeciesList();
        species->insert("S1");
        species->insert("S2");
        species->insert("S3");
        style->newColorMapper(this, "S1");
        registerType();
    };

};


int main(int argc, char const *argv[])
{
    BoundaryConditionsArgsContainer *bcArgs = new BoundaryConditionsArgsContainer();
    bcArgs->setValue("x", BOUNDARY_PERIODIC | BOUNDARY_RESETTING);

    Simulator::Config config;
    config.setWindowless(true);
    config.universeConfig.dt = 0.1;
    config.universeConfig.dim = {15, 6, 6};
    config.universeConfig.spaceGridSize = {9, 3, 3};
    config.universeConfig.cutoff = 3;
    config.universeConfig.setBoundaryConditions(bcArgs);
    TF_TEST_CHECK(init(config));

    AType *A = new AType();
    A = (AType*)A->get();

    Fluxes::flux(A, A, "S1", 2);

    ParticleHandle *a1, *a2;
    FVector3 pos;
    pos = Universe::getCenter() - FVector3(0, 1, 0);
    a1 = (*A)(&pos);
    pos = Universe::getCenter() + FVector3(-5, 1, 0);
    FVector3 vel(0.5, 0, 0);
    a2 = (*A)(&pos, &vel);

    a1->getSpecies()->setItem(a1->getSpecies()->species->index_of("S1"), 3.0);
    a2->getSpecies()->setItem(a2->getSpecies()->species->index_of("S1"), 0.0);

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
