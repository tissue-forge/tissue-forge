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

#ifndef _MDCORE_INCLUDE_TFRUNNER_H_
#define _MDCORE_INCLUDE_TFRUNNER_H_

#include "tf_platform.h"
#include "cycle.h"

/* runner error codes */
#define runner_err_ok                    0
#define runner_err_null                  -1
#define runner_err_malloc                -2
#define runner_err_space                 -3
#define runner_err_pthread               -4
#define runner_err_engine                -5
#define runner_err_spe                   -6
#define runner_err_mfc                   -7
#define runner_err_unavail               -8
#define runner_err_fifo                  -9
#define runner_err_verlet_overflow       -10
#define runner_err_tasktype              -11


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


	/* the last error */
	CAPI_DATA(int) runner_err;

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
	int runner_dopair_unsorted(struct runner *r, struct space_cell *cell_i, struct space_cell *cell_j);

	int runner_init(struct runner *r, struct engine *e, int id);
	int runner_run(struct runner *r);
	void runner_sort_ascending(unsigned int *parts, int N);
	void runner_sort_descending(unsigned int *parts, int N);
	int runner_verlet_eval(struct runner *r, struct space_cell *c, FPTYPE *f_out);
	int runner_verlet_fill(struct runner *r, struct space_cell *cell_i, struct space_cell *cell_j, FPTYPE *pshift);
	int runner_dosort(struct runner *r, struct space_cell *c, int flags);
	int runner_dopair(struct runner *r, struct space_cell *cell_i, struct space_cell *cell_j, int sid);
	int runner_doself(struct runner *r, struct space_cell *cell_i);

};

MDCORE_END_DECLS

#endif // _MDCORE_INCLUDE_TFRUNNER_H_