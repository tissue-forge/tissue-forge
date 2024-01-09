/*******************************************************************************
 * This file is part of Tissue Forge.
 * Copyright (c) 2023-2024 T.J. Sego
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
 * @file tfColorMaps.h
 * 
 */

#ifndef _SOURCE_RENDERING_TFCOLORMAPS_H_
#define _SOURCE_RENDERING_TFCOLORMAPS_H_

#include "tfStyle.h"

#include <string>
#include <vector>


namespace TissueForge::rendering {


    typedef fVector4 (*ColorMapperFunc)(struct ColorMapper *mapper, const float& s);

    ColorMapperFunc getColorMapperFunc(const std::string& name);

    std::vector<std::string> getColorMapperFuncNames();

};

#endif // _SOURCE_RENDERING_TFCOLORMAPS_H_