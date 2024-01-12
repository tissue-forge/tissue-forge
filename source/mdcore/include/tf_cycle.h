/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * Coypright (c) 2017 Andy Somogyi (somogyie at indiana dot edu)
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

/**
 * @file tf_cycle.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TF_CYCLE_H_
#define _MDCORE_INCLUDE_TF_CYCLE_H_

#if defined(TF_APPLE) && defined(__arm64__) && !defined(HAVE_TICK_COUNTER)
typedef unsigned long long ticks;
#define getticks() clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)
#define HAVE_TICK_COUNTER
#endif

#include "cycle.h"

#endif