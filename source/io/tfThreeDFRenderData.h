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

#ifndef _SOURCE_IO_TFTHREEDFRENDERDATA_H_
#define _SOURCE_IO_TFTHREEDFRENDERDATA_H_


#include <TissueForge_private.h>


namespace TissueForge::io {


    struct CAPI_EXPORT ThreeDFRenderData {

        FVector3 color = {0.f, 0.f, 0.f};

    };

};

#endif // _SOURCE_IO_TFTHREEDFRENDERDATA_H_