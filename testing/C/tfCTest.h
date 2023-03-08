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


#ifndef _TESTING_C_TFCTEST_H_
#define _TESTING_C_TFCTEST_H_

#include <stdlib.h>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <TissueForge_c.h>

#define TFC_TEST_CHECK(x) { if((x) != S_OK) { return E_FAIL; }; }


HRESULT tfTest_runQuiet(unsigned int numSteps) {
    tfFloatP_t dt;
    TFC_TEST_CHECK(tfUniverse_getDt(&dt));
    TFC_TEST_CHECK(tfStep(numSteps * dt, dt));
    return S_OK;
}


#endif // _TESTING_C_TFCTEST_H_