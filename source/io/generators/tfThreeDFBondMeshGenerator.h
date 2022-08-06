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

#ifndef _SOURCE_IO_GENERATORS_THREEDFBONDMESHGENERATOR_H_
#define _SOURCE_IO_GENERATORS_THREEDFBONDMESHGENERATOR_H_

#include <tfBond.h>

#include "tfThreeDFMeshGenerator.h"


namespace TissueForge::io {


    struct ThreeDFBondMeshGenerator : ThreeDFMeshGenerator { 

        /* Bonds of this mesh */
        std::vector<BondHandle> bonds;

        /* Mesh refinements applied when generating meshes from bonds */
        unsigned int pRefinements = 0;

        /* Radius of rendered bonds */
        FloatP_t radius = 0.01;
        
        // ThreeDFMeshGenerator interface
        
        HRESULT process();

    };

};

#endif // _SOURCE_IO_GENERATORS_THREEDFBONDMESHGENERATOR_H_