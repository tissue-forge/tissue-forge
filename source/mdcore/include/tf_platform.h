/*******************************************************************************
 * This file is part of mdcore.
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

#ifndef _MDCORE_INCLUDE_TF_PLATFORM_H_
#define _MDCORE_INCLUDE_TF_PLATFORM_H_

#include <tf_port.h>


#if defined(__cplusplus)
#define	MDCORE_BEGIN_DECLS	extern "C" {
#define	MDCORE_END_DECLS	}
#else
#define	MDCORE_BEGIN_DECLS
#define	MDCORE_END_DECLS
#endif

#endif // _MDCORE_INCLUDE_TF_PLATFORM_H_