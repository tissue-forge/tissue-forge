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

#include <limits.h>


using namespace TissueForge;


struct WellType : ParticleType {

    WellType() : ParticleType(true) {
        setFrozen(true);
        style->setVisible(false);
        registerType();
    };

};

struct SmallType : ParticleType {

    SmallType() : ParticleType(true) {
        radius = 0.1;
        setFrozenY(true);
        registerType();
    };

};


static FloatP_t eq_lam = -0.5;
static FloatP_t eq_mu = 1.0;
static int eq_s = 3;


static unsigned int factorial(const unsigned int &k) {
    unsigned int result = 1;
    for(unsigned int j = 1; j <= k; j++) 
        result *= j;
    return result;
}

static FloatP_t He(const FloatP_t &r, const unsigned int &n) {
    if(n == 0) return 1.0;
    else if(n == 1) return r;
    else return r * He(r, n - 1) - ((FloatP_t)n - 1.0) * He(r, n - 2);
}

static FloatP_t dgdr(const FloatP_t &_r, const unsigned int &n) {
    static FloatP_t eps = std::numeric_limits<FloatP_t>::epsilon();
    FloatP_t r = std::max(_r, eps);
    FloatP_t result = 0;
    for(int k = 1; k <= eq_s; k++) 
        if(2 * k - n >= 0) 
            result += (FloatP_t)factorial(2 * k) / (FloatP_t)factorial(2 * k - n) * (eq_lam + k) * std::pow(eq_mu, (FloatP_t)k) / (FloatP_t)factorial(k) * std::pow(r, 2.0 * k);
    return result / std::pow(r, (FloatP_t)n);
}

static FloatP_t u_n(const FloatP_t &r, const unsigned int &n) {
    return (FloatP_t)std::pow(-1, (int)n) * He(r, n) * eq_lam * exp(-eq_mu * pow(r, 2.0));
}

static FloatP_t f_n(const FloatP_t &r, const unsigned int &n) {
    FloatP_t w_n = 0.0;
    for(unsigned int j = 0; j <= n; j++) 
        w_n += (FloatP_t)factorial(n) / (FloatP_t)factorial(j) / (FloatP_t)factorial(n - j) * dgdr(r, j) * u_n(r, n - j);
    return 10.0 * (u_n(r, n) + w_n / eq_lam);
}

static FloatP_t f(FloatP_t r) {
    return f_n(r, 0);
}


int main(int argc, char const *argv[])
{
    BoundaryConditionsArgsContainer *bcArgs = new BoundaryConditionsArgsContainer();
    bcArgs->setValue("x", BOUNDARY_NO_SLIP);

    Simulator::Config config;
    config.setWindowless(true);
    config.universeConfig.cutoff = 5.0;
    config.universeConfig.setBoundaryConditions(bcArgs);
    TF_TEST_CHECK(tfTest_init(config));

    WellType *well_type = new WellType();
    SmallType *small_type = new SmallType();
    well_type = (WellType*)well_type->get();
    small_type = (SmallType*)small_type->get();

    Potential *pot_c = Potential::custom(0, 5, f, NULL, NULL);
    TF_TEST_CHECK(bind::types(pot_c, well_type, small_type));

    // Create particles
    FVector3 ucenter = Universe::getCenter();
    FVector3 udim = Universe::dim();
    FVector3 pos;
    (*well_type)(&ucenter);
    for(int i = 0; i < 20; i++) {
        pos = FVector3(FloatP_t(i + 1) / 21.0 * udim[0], ucenter[1], ucenter[2]);
        (*small_type)(&pos);
    }

    // run the simulator
    TF_TEST_CHECK(step(Universe::getDt() * 100));

    return S_OK;
}
