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

#ifndef _SOURCE_RENDERING_TFRENDERER_H_
#define _SOURCE_RENDERING_TFRENDERER_H_

#include <TissueForge_private.h>


namespace TissueForge {


    namespace rendering {


        struct Renderer 
        {
        };

        enum Renderer_Kind {
            RENDERER_WINDOWED               = 1 << 0,
            RENDERER_HEADLESS             = 1 << 1,
            
            RENDERER_WINDOWED_MAC           = (1 << 2) | (1 << 0),
            RENDERER_HEADLESS_MAC         = (1 << 2) | (1 << 1),
            
            RENDERER_WINDOWED_EGL           = (1 << 2) | (1 << 0),
            RENDERER_HEADLESS_EGL         = (1 << 2) | (1 << 1),
        
            RENDERER_WINDOWED_WINDOWS       = (1 << 2) | (1 << 0),
            RENDERER_HEADLESS_WINDOWS     = (1 << 2) | (1 << 1),
            
            RENDERER_WINDOWED_GLX           = (1 << 2) | (1 << 0),
            RENDERER_HEADLESS_GLX         = (1 << 2) | (1 << 1),
        };

        uint32_t availableRenderers();

}}

#endif // _SOURCE_RENDERING_TFRENDERER_H_