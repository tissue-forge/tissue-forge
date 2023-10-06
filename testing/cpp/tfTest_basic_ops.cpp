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
        registerType();
    };

};

struct BType : ParticleType {

    BType() : ParticleType(true) {
        registerType();
    };

};

struct CType : ClusterParticleType {

    CType() : ClusterParticleType(true) {
        AType A;
        types.insert((AType*)A.get());

        registerType();
    };
};

struct DType : ClusterParticleType {

    DType() : ClusterParticleType(true) {
        BType B;
        types.insert((BType*)B.get());

        registerType();
    };
};

template <typename S, typename T> 
HRESULT check_fnc(S op, T res) {
    if(op != res) { 
        std::cerr << "Failed! " << op << " " << res << std::endl;
        return E_FAIL;
    }
    return S_OK;
}


int main(int argc, char const *argv[])
{
    Simulator::Config config;
    config.setWindowless(true);
    TF_TEST_CHECK(tfTest_init(config));

    std::cout << "Test particle types:" << std::endl;

    AType *A = new AType();
    A = (AType*)A->get();
    std::cout << "\tA: " << *A << std::endl;

    BType *B = new BType();
    B = (BType*)B->get();
    std::cout << "\tB: " << *B << std::endl;

    CType *C = new CType();
    C = (CType*)C->get();
    std::cout << "\tC: " << *C << std::endl;

    DType *D = new DType();
    D = (DType*)D->get();
    std::cout << "\tD: " << *D << std::endl;


    // Make some test particles

    std::cout << "Test particles:" << std::endl;

    ParticleHandle *a = (*A)();
    std::cout << "\ta : " << *a << std::endl;

    ParticleHandle *b = (*B)();
    std::cout << "\tb : " << *b << std::endl;

    ClusterParticleHandle *c = (ClusterParticleHandle*)(*C)();
    std::cout << "\tc : " << *c << std::endl;

    ClusterParticleHandle *d = (ClusterParticleHandle*)(*D)();
    std::cout << "\td : " << *d << std::endl;

    ParticleHandle *c0 = (*c)(A);
    std::cout << "\tc0: " << *c0 << std::endl;

    ParticleHandle *d0 = (*d)(B);
    std::cout << "\td0: " << *d0 << std::endl;


    // Make some test bonds

    std::cout << "Test bonds:" << std::endl;

    Potential *pot_bond = Potential::linear(1);
    BondHandle *bond_ab = Bond::create(pot_bond, a, b);
    std::cout << "\tbond    : " << *bond_ab << std::endl;
    
    AngleHandle *angle_abc0 = Angle::create(pot_bond, a, b, c0);
    std::cout << "\tangle   : " << *angle_abc0 << std::endl;
    
    DihedralHandle *dihedral_abc0d0 = Dihedral::create(pot_bond, a, b, c0, d0);
    std::cout << "\tdihedral: " << *dihedral_abc0d0 << std::endl;
    
    
    // Container-like operations

    std::cout << "Container-like operations" << std::endl;

    TF_TEST_CHECK(check_fnc(A->parts.nr_parts, 2));
    TF_TEST_CHECK(check_fnc(B->parts.nr_parts, 2));
    TF_TEST_CHECK(check_fnc(c->items().nr_parts, 1));
    TF_TEST_CHECK(check_fnc(d->items().nr_parts, 1));
    TF_TEST_CHECK(check_fnc(A->has(a), true));
    TF_TEST_CHECK(check_fnc(A->has(b), false));
    TF_TEST_CHECK(check_fnc(B->has(b), true));
    TF_TEST_CHECK(check_fnc(C->has(A), true));
    TF_TEST_CHECK(check_fnc(C->has(B), false));
    TF_TEST_CHECK(check_fnc(D->has(A), false));
    TF_TEST_CHECK(check_fnc(D->has(B), true));
    TF_TEST_CHECK(check_fnc(c->has(c0), true));
    TF_TEST_CHECK(check_fnc(d->has(c0), false));
    TF_TEST_CHECK(check_fnc(c->has(d0), false));
    TF_TEST_CHECK(check_fnc(d->has(d0), true));

    TF_TEST_CHECK(check_fnc(bond_ab->has(a), true));
    TF_TEST_CHECK(check_fnc(bond_ab->has(b), true));
    TF_TEST_CHECK(check_fnc(bond_ab->has(c), false));
    TF_TEST_CHECK(check_fnc(bond_ab->has(d), false));
    TF_TEST_CHECK(check_fnc(bond_ab->has(c0), false));
    TF_TEST_CHECK(check_fnc(bond_ab->has(d0), false));

    TF_TEST_CHECK(check_fnc(angle_abc0->has(a), true));
    TF_TEST_CHECK(check_fnc(angle_abc0->has(b), true));
    TF_TEST_CHECK(check_fnc(angle_abc0->has(c), false));
    TF_TEST_CHECK(check_fnc(angle_abc0->has(d), false));
    TF_TEST_CHECK(check_fnc(angle_abc0->has(c0), true));
    TF_TEST_CHECK(check_fnc(angle_abc0->has(d0), false));

    TF_TEST_CHECK(check_fnc(dihedral_abc0d0->has(a), true));
    TF_TEST_CHECK(check_fnc(dihedral_abc0d0->has(b), true));
    TF_TEST_CHECK(check_fnc(dihedral_abc0d0->has(c), false));
    TF_TEST_CHECK(check_fnc(dihedral_abc0d0->has(d), false));
    TF_TEST_CHECK(check_fnc(dihedral_abc0d0->has(c0), true));
    TF_TEST_CHECK(check_fnc(dihedral_abc0d0->has(d0), true));

    TF_TEST_CHECK(check_fnc(Universe::particles().nr_parts, 6));
    TF_TEST_CHECK(check_fnc(bond_ab->getPartList().nr_parts, 2));
    TF_TEST_CHECK(check_fnc(angle_abc0->getPartList().nr_parts, 3));
    TF_TEST_CHECK(check_fnc(dihedral_abc0d0->getPartList().nr_parts, 4));

    std::cout << "\tPassed!" << std::endl;


    // Comparison operations

    std::cout << "Comparison operations" << std::endl;

    ParticleHandle aa(a->id);    // Make another handle to the first particle
    std::cout << "\tAnother handle to particle " << a->id << ": " << aa << std::endl;
    TF_TEST_CHECK(check_fnc(*a == aa, true));
    TF_TEST_CHECK(check_fnc(*a < *b, true));
    TF_TEST_CHECK(check_fnc(*a > *c, false));
    TF_TEST_CHECK(check_fnc(*A < *B, true));
    TF_TEST_CHECK(check_fnc(*A > *C, false));

    std::cout << "\tPassed!" << std::endl;

    return S_OK;
}
