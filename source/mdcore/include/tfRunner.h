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
 * @file tfRunner.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFRUNNER_H_
#define _MDCORE_INCLUDE_TFRUNNER_H_

#include "tf_platform.h"
#include "tf_cycle.h"


/* some constants */
/* Minimum number of nanoseconds to sleep if no task available. */
#define runner_minsleep                  1000

/** Maximum number of cellpairs to get from space_getpair. */
#define runner_bitesize                  3

/** Number of particles to request per call to space_getverlet. */
#define runner_verlet_bitesize           200

/** Magic word to make the dispatcher stop. */
#define runner_dispatch_stop             0xffffffff
#define runner_dispatch_lookahead        20


/** Timers. */
enum {
	runner_timer_queue = 0,
	runner_timer_pair,
	runner_timer_self,
	runner_timer_sort,
	runner_timer_count
};


namespace TissueForge {


	CAPI_DATA(ticks) runner_timers[];

};

#ifdef TIMER
#define TIMER_TIC_ND tic = getticks();
#define TIMER_TIC2_ND ticks tic2 = getticks();
#define TIMER_TIC ticks tic = getticks();
#define TIMER_TOC(t) timers_toc(t, tic)
#define TIMER_TIC2 ticks tic2 = getticks();
#define TIMER_TOC2(t) timers_toc(t, tic2)
#ifndef INLINE
# if __GNUC__ && !__GNUC_STDC_INLINE__
#  define INLINE extern inline
# else
#  define INLINE inline
# endif
#endif
INLINE static ticks timers_toc(int t, ticks tic) {
	ticks d = (getticks() - tic);
	__sync_add_and_fetch(&TissueForge::runner_timers[t], d);
	return d;
}
#else
#define TIMER_TIC_ND
#define TIMER_TIC
#define TIMER_TOC(t)
#define TIMER_TIC2
#define TIMER_TOC2(t)
#endif

MDCORE_BEGIN_DECLS


namespace TissueForge { 


	/* The fifo-queue for dispatching. */
	typedef struct runner_fifo {

		/* Access mutex and condition signal for blocking use. */
		pthread_mutex_t mutex;
		pthread_cond_t cond;

		/* Counters. */
		int first, last, size, count;

		/* The FIFO data. */
		int *data;

	} runner_fifo;

	/* the runner structure */
	typedef struct runner {

		/* the engine with which i am associated */
		struct engine *e;

		/* this runner's id */
		int id;

		/* my thread */
		pthread_t thread;

		/** ID of the last error on this runner. */
		int err;

		/** Accumulated potential energy by this runner. */
		FPTYPE epot;

	} runner;

	/* associated functions */

	/**
	 * @brief Initialize the runner associated to the given engine.
	 * 
	 * @param r The #runner to be initialized.
	 * @param e The #engine with which it is associated.
	 * @param id The ID of this #runner.
	 */
	HRESULT runner_init(struct runner *r, struct engine *e, int id);

	/** Run a #runner */
	HRESULT runner_run(struct runner *r);

	/**
	 * @brief Sort the particles in ascending order using QuickSort.
	 *
	 * @param parts The particle IDs and distances in compact form
	 * @param N The number of particles.
	 *
	 * The particle data is assumed to contain the distance in the lower
	 * 16 bits and the particle ID in the upper 16 bits.
	 */
	void runner_sort_ascending(unsigned int *parts, int N);

	/**
	 * @brief Sort the particles in descending order using QuickSort.
	 *
	 * @param parts The particle IDs and distances in compact form
	 * @param N The number of particles.
	 *
	 * The particle data is assumed to contain the distance in the lower
	 * 16 bits and the particle ID in the upper 16 bits.
	 */
	void runner_sort_descending(unsigned int *parts, int N);

	/**
	 * @brief Compute the interactions between the particles in the given
	 *        space_cell using the verlet list.
	 *
	 * @param r The #runner.
	 * @param c The #cell containing the particles to traverse.
	 * @param f A pointer to an array of #FPTYPE in which to aggregate the
	 *        interaction forces.
	 */
	HRESULT runner_verlet_eval(struct runner *r, struct space_cell *c, FPTYPE *f_out);

	/**
	 * @brief Fill in the Verlet list entries for the given space_cell pair.
	 * 
	 * @param r The #runner computing the pair.
	 * @param cell_i The first cell.
	 * @param cell_j The second cell.
	 * @param pshift A pointer to an array of three floating point values containing
	 *      the vector separating the centers of @c cell_i and @c cell_j.
	 */
	HRESULT runner_verlet_fill(struct runner *r, struct space_cell *cell_i, struct space_cell *cell_j, FPTYPE *pshift);

	/**
	 * @brief Fill in the pairwise Verlet list entries for the given cell pair
	 *        if needed and compute the interactions.
	 * 
	 * @param r The #runner computing the pair.
	 * @param c The cell.
	 * @param flags Bitmask for the sorting directions.
	 *
	 * This routine differs from #runner_dopair_verlet in that instead of
	 * storing a Verlet table, the sorted particle ids are stored. This
	 * requires only (size_i + size_j) entries as opposed to size_i*size_j
	 * for the Verlet table, yet may be less efficient since particles
	 * within the skin along the cell-pair axis are inspected, as opposed
	 * to particles simply within the skin of each other.
	 *
	 */
	HRESULT runner_dosort(struct runner *r, struct space_cell *c, int flags);

	/**
	 * @brief Compute the pairwise interactions for the given pair.
	 *
	 * @param r The #runner computing the pair.
	 * @param cell_i The first cell.
	 * @param cell_j The second cell.
	 * @param shift A pointer to an array of three floating point values containing
	 *      the vector separating the centers of @c cell_i and @c cell_j.
	 *
	 * Computes the interactions between all the particles in @c cell_i and all
	 * the particles in @c cell_j. @c cell_i and @c cell_j may be the same cell.
	 *
	 * @sa #runner_sortedpair.
	 */
	HRESULT runner_dopair(struct runner *r, struct space_cell *cell_i, struct space_cell *cell_j, int sid);

	/**
	 * @brief Compute the pairwise fluxes for the given pair.
	 *
	 * @param r The #runner computing the pair.
	 * @param cell_i The first cell.
	 * @param cell_j The second cell.
	 * @param shift A pointer to an array of three floating point values containing
	 *      the vector separating the centers of @c cell_i and @c cell_j.
	 *
	 * Computes the fluxes between all the particles in @c cell_i and all
	 * the particles in @c cell_j. @c cell_i and @c cell_j may be the same cell.
	 *
	 * @sa #runner_sortedpair.
	 */
	HRESULT runner_dopair_fluxonly(struct runner *r, struct space_cell *cell_i, struct space_cell *cell_j, int sid);

	/**
	 * @brief Compute the self-interactions for the given cell.
	 *
	 * @param r The #runner computing the pair.
	 * @param cell_i The first cell.
	 */
	HRESULT runner_doself(struct runner *r, struct space_cell *cell_i);

	/**
	 * @brief Compute the self-fluxes for the given cell.
	 *
	 * @param r The #runner computing the pair.
	 * @param cell_i The first cell.
	 */
	HRESULT runner_doself_fluxonly(struct runner *r, struct space_cell *cell_i);

};

MDCORE_END_DECLS

#endif // _MDCORE_INCLUDE_TFRUNNER_H_