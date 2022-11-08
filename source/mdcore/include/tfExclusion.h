/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * Coypright (c) 2017 Andy Somogyi (somogyie at indiana dot edu)
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
 * @file tfExclusion.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFEXCLUSION_H_
#define _MDCORE_INCLUDE_TFEXCLUSION_H_
#include "tf_platform.h"

MDCORE_BEGIN_DECLS


namespace TissueForge { 


	/** The exclusion structure */
	typedef struct exclusion {

		/* ids of particles involved */
		int i, j;

	} exclusion;


	/* associated functions */

	/**
	 * @brief Evaluate a list of exclusioned interactions
	 *
	 * @param b Pointer to an array of #exclusion.
	 * @param N Nr of exclusions in @c b.
	 * @param e Pointer to the #engine in which these exclusions are evaluated.
	 * @param epot_out Pointer to a FPTYPE in which to aggregate the potential energy.
	 */
	HRESULT exclusion_eval(struct exclusion *b, int N, struct engine *e, FPTYPE *epot_out);

	/**
	 * @brief Evaluate a list of exclusioned interactions
	 *
	 * @param b Pointer to an array of #exclusion.
	 * @param N Nr of exclusions in @c b.
	 * @param e Pointer to the #engine in which these exclusions are evaluated.
	 * @param f Array in which to aggregate the force.
	 * @param epot_out Pointer to a FPTYPE in which to aggregate the potential energy.
	 */
	HRESULT exclusion_evalf(struct exclusion *b, int N, struct engine *e, FPTYPE *f, FPTYPE *epot_out);

};

MDCORE_END_DECLS
#endif // _MDCORE_INCLUDE_TFEXCLUSION_H_