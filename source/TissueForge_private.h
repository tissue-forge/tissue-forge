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

#ifndef _SOURCE_TISSUEFORGE_PRIVATE_H_
#define _SOURCE_TISSUEFORGE_PRIVATE_H_

#include <tf_port.h>
#include <tf_style.h>

// Setup for importing numpy and setting up function pointers.
#ifndef TF_IMPORTING_NUMPY_ARRAY
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL TISSUEFORGE_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include <numpy/arrayobject.h>

#include <assert.h>
#include <algorithm>

#define TF_NOTIMPLEMENTED \
    assert("Not Implemented" && 0);\
    return 0;

#define TF_NOTIMPLEMENTED_NORET assert("Not Implemented" && 0);

#include "types/tf_types.h"
#include "types/tf_cast.h"

#endif // _SOURCE_TISSUEFORGE_PRIVATE_H_