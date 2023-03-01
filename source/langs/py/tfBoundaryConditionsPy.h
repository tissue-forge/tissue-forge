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

/**
 * @file tfBoundaryConditionsPy.h
 * 
 */

#ifndef _SOURCE_LANGS_PY_TFBOUNDARYCONDITIONSPY_H_
#define _SOURCE_LANGS_PY_TFBOUNDARYCONDITIONSPY_H_

#include "tf_py.h"
#include <tfBoundaryConditions.h>
#include <tfSpace.h>


namespace TissueForge::py {


    enum class BoundaryTypeFlags : int {
        BOUNDARY_NONE = space_periodic_none,
        PERIODIC_X = space_periodic_x,
        PERIODIC_Y = space_periodic_y,
        PERIODIC_Z = space_periodic_z,
        PERIODIC_FULL = space_periodic_full,
        PERIODIC_GHOST_X = space_periodic_ghost_x,
        PERIODIC_GHOST_Y = space_periodic_ghost_y,
        PERIODIC_GHOST_Z = space_periodic_ghost_z,
        PERIODIC_GHOST_FULL = space_periodic_ghost_full,
        FREESLIP_X = SPACE_FREESLIP_X,
        FREESLIP_Y = SPACE_FREESLIP_Y,
        FREESLIP_Z = SPACE_FREESLIP_Z,
        FREESLIP_FULL = SPACE_FREESLIP_FULL
    };

    struct CAPI_EXPORT BoundaryConditionsArgsContainerPy : BoundaryConditionsArgsContainer {

        BoundaryConditionsArgsContainerPy(PyObject *obj);

    };

};

#endif // _SOURCE_LANGS_PY_TFBOUNDARYCONDITIONSPY_H_