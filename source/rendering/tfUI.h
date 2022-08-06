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

/**
 * @file tfUI.h
 * 
 */

#ifndef _SOURCE_RENDERING_TFUI_H_
#define _SOURCE_RENDERING_TFUI_H_

#include <TissueForge_private.h>


namespace TissueForge::rendering {


    CAPI_FUNC(HRESULT) pollEvents();
    CAPI_FUNC(HRESULT) waitEvents(double timeout);
    CAPI_FUNC(HRESULT) postEmptyEvent();
    CAPI_FUNC(HRESULT) initializeGraphics();

}

#endif // _SOURCE_RENDERING_TFUI_H_