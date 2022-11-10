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

#include "tfCTest.h"


#define check_fnc(op, res) { if(op != res) { fprintf(stderr, "Failed!"); return E_FAIL; } else { return S_OK; } }


int main(int argc, char** argv) {

    struct tfSimulatorConfigHandle config;
    TFC_TEST_CHECK(tfSimulatorConfig_init(&config));
    TFC_TEST_CHECK(tfSimulatorConfig_setWindowless(&config, 1));
    TFC_TEST_CHECK(tfInitC(&config, NULL, 0));

    struct tfParticleTypeHandle A, B;
    struct tfClusterParticleTypeHandle C, D;

    printf("Test particle types:\n");

    char *str;
    unsigned int numChars;

    TFC_TEST_CHECK(tfParticleType_init(&A));
    TFC_TEST_CHECK(tfParticleType_setName(&A, "AType"));
    TFC_TEST_CHECK(tfParticleType_registerType(&A));
    TFC_TEST_CHECK(tfParticleType_str(&A, &str, &numChars));
    printf("\tA: %s\n", str);
    free(str);

    TFC_TEST_CHECK(tfParticleType_init(&B));
    TFC_TEST_CHECK(tfParticleType_setName(&B, "BType"));
    TFC_TEST_CHECK(tfParticleType_registerType(&B));
    TFC_TEST_CHECK(tfParticleType_str(&B, &str, &numChars));
    printf("\tB: %s\n", str);
    free(str);

    TFC_TEST_CHECK(tfClusterParticleType_init(&C));
    TFC_TEST_CHECK(tfParticleType_setName((struct tfParticleTypeHandle*)(&C), "CType"));
    TFC_TEST_CHECK(tfClusterParticleType_addType(&C, &A));
    TFC_TEST_CHECK(tfClusterParticleType_registerType(&C));
    TFC_TEST_CHECK(tfClusterParticleType_str(&C, &str, &numChars));
    printf("\tC: %s\n", str);
    free(str);

    TFC_TEST_CHECK(tfClusterParticleType_init(&D));
    TFC_TEST_CHECK(tfParticleType_setName((struct tfParticleTypeHandle*)(&D), "DType"));
    TFC_TEST_CHECK(tfClusterParticleType_addType(&D, &B));
    TFC_TEST_CHECK(tfClusterParticleType_registerType(&D));
    TFC_TEST_CHECK(tfClusterParticleType_str(&D, &str, &numChars));
    printf("\tD: %s\n", str);
    free(str);


    // Make some test particles

    struct tfParticleHandleHandle a, b, c0, d0;
    struct tfClusterParticleHandleHandle c, d;
    int pid;

    TFC_TEST_CHECK(tfParticleType_createParticle(&A, &pid, NULL, NULL));
    TFC_TEST_CHECK(tfParticleHandle_init(&a, pid));
    TFC_TEST_CHECK(tfParticleHandle_str(&a, &str, &numChars));
    printf("\ta: %s\n", str);
    free(str);
    
    TFC_TEST_CHECK(tfParticleType_createParticle(&B, &pid, NULL, NULL));
    TFC_TEST_CHECK(tfParticleHandle_init(&b, pid));
    TFC_TEST_CHECK(tfParticleHandle_str(&b, &str, &numChars));
    printf("\tb: %s\n", str);
    free(str);
    
    TFC_TEST_CHECK(tfClusterParticleType_createParticle(&C, &pid, NULL, NULL));
    TFC_TEST_CHECK(tfClusterParticleHandle_init(&c, pid));
    TFC_TEST_CHECK(tfClusterParticleHandle_str(&c, &str, &numChars));
    printf("\tc: %s\n", str);
    free(str);
    
    TFC_TEST_CHECK(tfClusterParticleType_createParticle(&D, &pid, NULL, NULL));
    TFC_TEST_CHECK(tfClusterParticleHandle_init(&d, pid));
    TFC_TEST_CHECK(tfClusterParticleHandle_str(&d, &str, &numChars));
    printf("\td: %s\n", str);
    free(str);
    
    TFC_TEST_CHECK(tfClusterParticleHandle_createParticle(&c, &A, &pid, NULL, NULL));
    TFC_TEST_CHECK(tfParticleHandle_init(&c0, pid));
    TFC_TEST_CHECK(tfParticleHandle_str(&c0, &str, &numChars));
    printf("\tc0: %s\n", str);
    free(str);
    
    TFC_TEST_CHECK(tfClusterParticleHandle_createParticle(&d, &B, &pid, NULL, NULL));
    TFC_TEST_CHECK(tfParticleHandle_init(&d0, pid));
    TFC_TEST_CHECK(tfParticleHandle_str(&d0, &str, &numChars));
    printf("\td0: %s\n", str);
    free(str);
    
    // Make some test bonds

    struct tfPotentialHandle pot_bond;
    struct tfBondHandleHandle bond_ab;
    struct tfAngleHandleHandle angle_abc0;
    struct tfDihedralHandleHandle dihedral_abc0d0;
    TFC_TEST_CHECK(tfPotential_create_linear(&pot_bond, 1, NULL, NULL, NULL));
    TFC_TEST_CHECK(tfBondHandle_create(&bond_ab, &pot_bond, &a, &b));
    TFC_TEST_CHECK(tfAngleHandle_create(&angle_abc0, &pot_bond, &a, &b, &c0));
    TFC_TEST_CHECK(tfDihedralHandle_create(&dihedral_abc0d0, &pot_bond, &a, &b, &c0, &d0));

    printf("Test bonds:\n");
    TFC_TEST_CHECK(tfBondHandle_str(&bond_ab, &str, &numChars));
    printf("\tbond    : %s\n", str);
    free(str);
    TFC_TEST_CHECK(tfAngleHandle_str(&angle_abc0, &str, &numChars));
    printf("\tangle   : %s\n", str);
    free(str);
    TFC_TEST_CHECK(tfDihedralHandle_str(&dihedral_abc0d0, &str, &numChars));
    printf("\tdihedral: %s\n", str);
    free(str);

    // Container-like operations

    printf("Container-like operations\n");

    int numParts;

    TFC_TEST_CHECK(tfParticleType_getNumParts(&A, &numParts));
    check_fnc(numParts, 2);
    TFC_TEST_CHECK(tfParticleType_getNumParts(&B, &numParts));
    check_fnc(numParts, 2);
    TFC_TEST_CHECK(tfClusterParticleHandle_getNumParts(&c, &numParts));
    check_fnc(numParts, 1);
    TFC_TEST_CHECK(tfClusterParticleHandle_getNumParts(&d, &numParts));
    check_fnc(numParts, 1);

    bool result;

    TFC_TEST_CHECK(tfParticleType_hasPart(&A, &a, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfParticleType_hasPart(&A, &b, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfParticleType_hasPart(&B, &b, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfClusterParticleType_hasType(&C, &A, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfClusterParticleType_hasType(&C, &B, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfClusterParticleType_hasType(&D, &A, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfClusterParticleType_hasType(&D, &B, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfClusterParticleHandle_hasPart(&c, &c0, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfClusterParticleHandle_hasPart(&d, &c0, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfClusterParticleHandle_hasPart(&c, &d0, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfClusterParticleHandle_hasPart(&d, &d0, &result));
    check_fnc(result, true);

    TFC_TEST_CHECK(tfBondHandle_hasPart(&bond_ab, &a, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfBondHandle_hasPart(&bond_ab, &b, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfBondHandle_hasPart(&bond_ab, &c, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfBondHandle_hasPart(&bond_ab, &d, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfBondHandle_hasPart(&bond_ab, &c0, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfBondHandle_hasPart(&bond_ab, &d0, &result));
    check_fnc(result, false);

    TFC_TEST_CHECK(tfAngleHandle_hasPart(&angle_abc0, &a, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfAngleHandle_hasPart(&angle_abc0, &b, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfAngleHandle_hasPart(&angle_abc0, &c, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfAngleHandle_hasPart(&angle_abc0, &d, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfAngleHandle_hasPart(&angle_abc0, &c0, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfAngleHandle_hasPart(&angle_abc0, &d0, &result));
    check_fnc(result, false);

    TFC_TEST_CHECK(tfDihedralHandle_hasPart(&dihedral_abc0d0, &a, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfDihedralHandle_hasPart(&dihedral_abc0d0, &b, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfDihedralHandle_hasPart(&dihedral_abc0d0, &c, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfDihedralHandle_hasPart(&dihedral_abc0d0, &d, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfDihedralHandle_hasPart(&dihedral_abc0d0, &c0, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfDihedralHandle_hasPart(&dihedral_abc0d0, &d0, &result));
    check_fnc(result, true);

    TFC_TEST_CHECK(tfUniverse_getNumParts(&numParts));
    check_fnc(numParts, 6);

    struct tfParticleListHandle plist;
    
    TFC_TEST_CHECK(tfBondHandle_getPartList(&bond_ab, &plist));
    TFC_TEST_CHECK(tfParticleList_getNumParts(&plist, &numParts));
    check_fnc(numParts, 2);
    TFC_TEST_CHECK(tfParticleList_destroy(&plist));
    
    TFC_TEST_CHECK(tfAngleHandle_getPartList(&angle_abc0, &plist));
    TFC_TEST_CHECK(tfParticleList_getNumParts(&plist, &numParts));
    check_fnc(numParts, 3);
    TFC_TEST_CHECK(tfParticleList_destroy(&plist));
    
    TFC_TEST_CHECK(tfDihedralHandle_getPartList(&dihedral_abc0d0, &plist));
    TFC_TEST_CHECK(tfParticleList_getNumParts(&plist, &numParts));
    check_fnc(numParts, 4);
    TFC_TEST_CHECK(tfParticleList_destroy(&plist));

    printf("\tPassed!\n");


    // Comparison operations

    printf("Comparison operations\n");

    // Make another handle to the first particle
    struct tfParticleHandleHandle aa;
    int a_id;
    TFC_TEST_CHECK(tfParticleHandle_getId(&a, &a_id));
    TFC_TEST_CHECK(tfParticleHandle_init(&aa, a_id));
    TFC_TEST_CHECK(tfParticleHandle_str(&aa, &str, &numChars));
    printf("\tAnother handle to particle %d: %s\n", a_id, str);
    free(str);

    TFC_TEST_CHECK(tfParticleHandle_eq(&a, &aa, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfParticleHandle_lt(&a, &b, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfParticleHandle_gt(&a, &c, &result));
    check_fnc(result, false);
    TFC_TEST_CHECK(tfParticleType_lt(&A, &B, &result));
    check_fnc(result, true);
    TFC_TEST_CHECK(tfParticleType_gt(&A, &C, &result));
    check_fnc(result, false);

    printf("\tPassed!\n");

    return 0;
}