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

#include <string>


using namespace TissueForge;


static int colorCount;


struct BeadType : ParticleType {

    int i;

    BeadType() : ParticleType(true) {
        radius = 3.0;
        species = new state::SpeciesList();
        species->insert("S1");
        style->mapper = new rendering::ColorMapper();
        style->mapper->setMapParticleSpecies(this, "S1");
        registerType();
    };

};


HRESULT keypress(event::KeyEvent *e) {
    std::vector<std::string> mapperNames = rendering::ColorMapper::getNames();

    if(std::strcmp(e->keyName().c_str(), "n")) {
        colorCount = (colorCount + 1) % mapperNames.size();
    } 
    else if(std::strcmp(e->keyName().c_str(), "p")) {
        colorCount = (colorCount - 1) % mapperNames.size();
    } 
    else {
        return S_OK;
    }

    ParticleType *Bead = BeadType().get();
    Bead->style->setColorMap(mapperNames[colorCount]);
    return S_OK;
};


int main(int argc, char const *argv[])
{
    Simulator::Config config;
    config.setWindowless(true);
    TF_TEST_CHECK(tfTest_init(config));

    BeadType *Bead = new BeadType();
    Bead = (BeadType*)Bead->get();

    std::vector<FVector3> pts = randomPoints(PointsType::Ring, 100);
    for(auto &p : pts) {
        FVector3 pt = p * 4 + Universe::getCenter();
        (*Bead)(&pt);
    }

    colorCount = 0;
    event::KeyEventHandlerType cb = &keypress;
    event::KeyEvent::addHandler(&cb);

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
