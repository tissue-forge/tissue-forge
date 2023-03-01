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

#ifndef _SOURCE_RENDERING_TFWINDOWLESS_H_
#define _SOURCE_RENDERING_TFWINDOWLESS_H_


#include <TissueForge.h>
#include <Magnum/GL/Context.h>

#if defined(TF_APPLE)
    #include "Magnum/Platform/WindowlessCglApplication.h"
#elif defined(TF_LINUX)
    #include "Magnum/Platform/WindowlessEglApplication.h"
#elif defined(TF_WINDOWS)
#include "Magnum/Platform/WindowlessWglApplication.h"
#else
#error no windowless application available on this platform
#endif



#endif // _SOURCE_RENDERING_TFWINDOWLESS_H_