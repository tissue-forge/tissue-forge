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


std::vector<std::string> speciesNames = {"S1", "S2", "S3"};


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

struct ProducerType : ParticleType {

    ProducerType() : ParticleType(true) {
        radius = 0.1;
        species = new state::SpeciesList();
        for(auto &s : speciesNames) species->insert(s);
        style->mapper = new rendering::ColorMapper();
        style->mapper->setMapParticleSpecies(this, "S1");
        registerType();
    };

};

struct ConsumerType : ParticleType {

    ConsumerType() : ParticleType(true) {
        radius = 0.1;
        species = new state::SpeciesList();
        for(auto &s : speciesNames) species->insert(s);
        style->newColorMapper(this, "S1");
        registerType();
    };

};


int main(int argc, char const *argv[])
{
    BoundaryConditionsArgsContainer *bcArgs = new BoundaryConditionsArgsContainer();
    bcArgs->setValueAll(BOUNDARY_FREESLIP);

    Simulator::Config config;
    config.universeConfig.dim = {6.5, 6.5, 6.5};
    config.universeConfig.setBoundaryConditions(bcArgs);
    config.setWindowless(true);
    TF_TEST_CHECK(init(config));

    AType *A = new AType();
    ProducerType *Producer = new ProducerType();
    ConsumerType *Consumer = new ConsumerType();
    A = (AType*)A->get();
    Producer = (ProducerType*)Producer->get();
    Consumer = (ConsumerType*)Consumer->get();

    // define fluxes between objects types
    Fluxes::flux(A, A, "S1", 5.0, 0.0);
    Fluxes::secrete(Producer, A, "S1", 5.0, 0.0);
    Fluxes::uptake(A, Consumer, "S1", 10.0, 500.0);

    // make a bunch of objects
    A->factory(10000);

    // Grab some objects
    ParticleHandle *producer = A->parts.item(0);
    ParticleHandle *consumer = A->parts.item(A->parts.nr_parts - 1);

    // Change types
    TF_TEST_CHECK(producer->become(Producer));
    TF_TEST_CHECK(consumer->become(Consumer));

    // Set initial condition
    producer->getSpecies()->setItem(producer->getSpecies()->species->index_of("S1"), 2000.0);

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
