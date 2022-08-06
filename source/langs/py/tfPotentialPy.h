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

#ifndef _SOURCE_LANGS_PY_TFPOTENTIALPY_H_
#define _SOURCE_LANGS_PY_TFPOTENTIALPY_H_

#include "tf_py.h"
#include <tfPotential.h>


namespace TissueForge::py {


    struct CAPI_EXPORT PotentialPy : Potential {
        /**
         * @brief Creates a custom potential. 
         * 
         * @param min The smallest radius for which the potential will be constructed.
         * @param max The largest radius for which the potential will be constructed.
         * @param f function returning the value of the potential
         * @param fp function returning the value of first derivative of the potential
         * @param f6p function returning the value of sixth derivative of the potential
         * @param tol Tolerance, defaults to 0.001.
         * @return Potential* 
         */
        static Potential *customPy(
            FloatP_t min, 
            FloatP_t max, 
            PyObject *f, 
            PyObject *fp=Py_None, 
            PyObject *f6p=Py_None, 
            FloatP_t *tol=NULL, 
            uint32_t *flags=NULL
        );
    };

};

#endif // _SOURCE_LANGS_PY_TFPOTENTIALPY_H_