/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <tfCluster.h>
#include <tfFlux.h>

/* Include conditional headers. */
#include <mdcore_config.h>
#ifdef WITH_MPI
#include <mpi.h>
#endif
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#ifdef WITH_METIS
#include <metis.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif

/* include local headers */
#include <cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfParticle.h>
#include <tfSpace_cell.h>
#include <tfTask.h>
#include <tfQueue.h>
#include <tfSpace.h>
#include <tfPotential.h>
#include <tfRunner.h>
#include <tfBond.h>
#include <tfRigid.h>
#include <tfAngle.h>
#include <tfDihedral.h>
#include <tfExclusion.h>
#include <tfEngine.h>
#include "tf_engine_advance.h"
#include <tfForce.h>
#include <tfBoundaryConditions.h>
#include <tfTaskScheduler.h>
#include <tfLogger.h>
#include <tf_util.h>
#include <tfError.h>
#include <iostream>

#pragma clang diagnostic ignored "-Wwritable-strings"


using namespace TissueForge;


#define error(id)				(tf_error(E_FAIL, errs_err_msg[id]))


/** TODO, clean up this design for types and static engine. */
/** What is the maximum nr of types? */
const int engine::max_type = engine_maxnrtypes;
int engine::nr_types = 0;

/**
 * The particle types.
 *
 * Currently initialized in _Particle_init
 */
ParticleType *engine::types = NULL;

static int init_types = 0;

static const unsigned steps_per_second_buffer_size = 10;
static FPTYPE steps_per_second_buffer[steps_per_second_buffer_size] = {0.0};
static FPTYPE steps_per_second_last = 0;
static unsigned steps_per_second_buffer_i = 0;

static void update_steps_per_second() {
    unsigned next = (steps_per_second_buffer_i + 1) % steps_per_second_buffer_size;
    FPTYPE time = util::wallTime();
    steps_per_second_buffer[steps_per_second_buffer_i] = time - steps_per_second_last;
    steps_per_second_last = time;
    steps_per_second_buffer_i = next;
}

FPTYPE TissueForge::engine_steps_per_second() {
    FPTYPE time = 0;
    for(int i = 0; i < steps_per_second_buffer_size-1; i++) {
        time += steps_per_second_buffer[i];
    }
    return steps_per_second_buffer_size / time;
}

HRESULT TissueForge::engine_shuffle(struct engine *e) {

	TF_Log(LOG_TRACE);

	struct space *s = &e->s;

#ifdef HAVE_OPENMP
	int cid, k;
	struct space_cell *c;
#endif

	/* Flush the ghost cells (to avoid overlapping particles) */
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static), private(cid)
    for(cid = 0 ; cid < s->nr_ghost ; cid++) {
		space_cell_flush(&(s->cells[s->cid_ghost[cid]]), s->partlist, s->celllist);
    }
#else
	auto func_space_cell_flush = [&](int _cid) {
		space_cell_flush(&(s->cells[s->cid_ghost[_cid]]), s->partlist, s->celllist);
    };
	parallel_for(s->nr_ghost, func_space_cell_flush);
#endif

	/* Shuffle the domain. */
    if(space_shuffle_local(s) != S_OK) {
		return error(MDCERR_space);
    }

#ifdef WITH_MPI
	/* Get the incomming particle from other procs if needed. */
	if(e->particle_flags & engine_flag_mpi)
		if(engine_exchange_incomming(e) != S_OK)
			return error(MDCERR_mpi);
#endif

/* Welcome the new particles in each cell, unhook the old ones. */
#ifdef HAVE_OPENMP
#pragma omp parallel for schedule(static), private(cid,c,k)
	for(cid = 0 ; cid < s->nr_marked ; cid++) {
		c = &(s->cells[s->cid_marked[cid]]);
		if(!(c->flags & cell_flag_ghost))
			space_cell_welcome(c, s->partlist);
		else {
			for(k = 0 ; k < c->incomming_count ; k++)
				e->s.partlist[ c->incomming[k].id ] = NULL;
			c->incomming_count = 0;
		}
	}
#else
	auto func_space_cell_welcome = [&](int _cid) {
		space_cell *_c = &(s->cells[s->cid_marked[_cid]]);
		if(!(_c->flags & cell_flag_ghost))
			space_cell_welcome(_c, s->partlist);
		else {
			for(int _k = 0 ; _k < _c->incomming_count ; _k++)
				s->partlist[ _c->incomming[_k].id ] = NULL;
			_c->incomming_count = 0;
		}
	};
	parallel_for(s->nr_marked, func_space_cell_welcome);
#endif

	TF_Log(LOG_TRACE);

	/* return quietly */
	return S_OK;

}


HRESULT TissueForge::engine_timers_reset(struct engine *e) {
    
    e->wall_time = 0;

	int k;

	/* Check input nonsense. */
	if(e == NULL)
		return error(MDCERR_null);

	/* Run through the timers and set them to 0. */
	for(k = 0 ; k < engine_timer_last ; k++)
		e->timers[k] = 0;

	/* What, that's it? */
	return S_OK;

}


HRESULT TissueForge::engine_verlet_update(struct engine *e) {

	int cid;
	FPTYPE maxdx = 0.0, skin;
	struct Particle *p;
	struct space *s = &e->s;
	ticks tic;
#ifdef HAVE_OPENMP
	int step, pid, k;
	FPTYPE lmaxdx, dx, w;
	struct space_cell *c;
#endif

	/* Do we really need to do this? */
	if(!(e->flags & engine_flag_verlet))
		return S_OK;

	/* Get the skin width. */
	skin = fmin(s->h[0], fmin(s->h[1], s->h[2])) - s->cutoff;

	/* Get the maximum particle movement. */
	if(!s->verlet_rebuild) {

#ifdef HAVE_OPENMP
#pragma omp parallel private(c,cid,pid,p,dx,k,w,step,lmaxdx)
		{
			lmaxdx = 0.0; step = omp_get_num_threads();
			for(cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step) {
				c = &(s->cells[s->cid_real[cid]]);
				for(pid = 0 ; pid < c->count ; pid++) {
					p = &(c->parts[pid]);
					for(dx = 0.0, k = 0 ; k < 3 ; k++) {
						w = p->x[k] - c->oldx[ 4*pid + k ];
						dx += w*w;
					}
					lmaxdx = fmax(dx, lmaxdx);
				}
			}
#pragma omp critical
			maxdx = fmax(lmaxdx, maxdx);
		}
#else

		std::vector<FPTYPE> wmaxdx(s->nr_real);
		auto func = [&](int _cid) {
			wmaxdx[_cid] = 0;
			FPTYPE _dx;
            space_cell *_c = &(s->cells[s->cid_real[_cid]]);
            for(int _pid = 0 ; _pid < _c->count ; _pid++) {
                Particle *_p = &(_c->parts[_pid]);
                for(int _k = 0, _dx = 0.0 ; _k < 3 ; _k++) {
                    FPTYPE _w = _p->x[_k] - _c->oldx[ 4*_pid + _k ];
                    _dx += _w * _w;
                }
                wmaxdx[_cid] = fmax(_dx, wmaxdx[_cid]);
            }
        };
		parallel_for(s->nr_real, func);
		for(cid = 0; cid < s->nr_real; cid++) 
			maxdx = fmax(maxdx, wmaxdx[cid]);

#endif

#ifdef WITH_MPI
/* Collect the maximum displacement from other nodes. */
if((e->particle_flags & engine_flag_mpi) &&(e->nr_nodes > 1)) {
	/* Do not use in-place as it is buggy when async is going on in the background. */
	if(MPI_Allreduce(MPI_IN_PLACE, &maxdx, 1, MPI_DOUBLE, MPI_MAX, e->comm) != MPI_SUCCESS)
		return error(MDCERR_mpi);
}
#endif

        /* Are we still in the green? */
        maxdx = sqrt(maxdx);
        s->verlet_rebuild =(2.0*maxdx > skin);

	}

	/* Do we have to rebuild the Verlet list? */
	if(s->verlet_rebuild) {

		/* Wait for any unterminated exchange. */
		tic = getticks();
#ifdef WITH_MPI
		if(e->particle_flags & engine_flag_async)
			if(engine_exchange_wait(e) != S_OK)
				return error(MDCERR_mpi);
#endif
        tic = getticks() - tic;
        e->timers[engine_timer_exchange1] += tic;
        e->timers[engine_timer_verlet] -= tic;

        /* Move the particles to the respecitve cells. */
        if(engine_shuffle(e) != S_OK)
            return error(MDCERR_engine);

        /* Store the current positions as a reference. */
#ifdef HAVE_OPENMP
        #pragma omp parallel for schedule(static), private(cid,c,pid,p,k)
        for(cid = 0 ; cid < s->nr_real ; cid++) {
            c = &(s->cells[s->cid_real[cid]]);
            if(c->oldx == NULL || c->oldx_size < c->count) {
                free(c->oldx);
                c->oldx_size = c->size + 20;
                c->oldx = (FPTYPE *)malloc(sizeof(FPTYPE) * 4 * c->oldx_size);
            }
            for(pid = 0 ; pid < c->count ; pid++) {
                p = &(c->parts[pid]);
                for(k = 0 ; k < 3 ; k++)
                    c->oldx[ 4*pid + k ] = p->x[k];
            }
        }
#else
		auto func_store_current_pos = [&](int _cid) -> void {
            space_cell *_c = &(s->cells[s->cid_real[_cid]]);
            if(_c->oldx == NULL || _c->oldx_size < _c->count) {
                free(_c->oldx);
                _c->oldx_size = _c->size + 20;
                _c->oldx = (FPTYPE *)malloc(sizeof(FPTYPE) * 4 * _c->oldx_size);
            }
            for(int _pid = 0 ; _pid < _c->count ; _pid++) {
                Particle *_p = &(_c->parts[_pid]);
                for(int _k = 0 ; _k < 3 ; _k++)
                    _c->oldx[ 4*_pid + _k ] = _p->x[_k];
            }
        };
		parallel_for(s->nr_real, func_store_current_pos);
#endif
        /* Set the maximum displacement to zero. */
        s->maxdx = 0;

	}

	/* Otherwise, just store the maximum displacement. */
	else
		s->maxdx = maxdx;

	/* All done! */
	return S_OK;

}

HRESULT TissueForge::engine_split(struct engine *e) {

	int i, k, cid, cjd;
	struct space_cell *ci, *cj, *ct;
	struct space *s = &(e->s);

	/* Check for nonsense inputs. */
	if(e == NULL)
		return error(MDCERR_null);

	/* Start by allocating and initializing the send/recv lists. */
	if((e->send = (struct engine_comm *)malloc(sizeof(struct engine_comm) * e->nr_nodes)) == NULL ||
			(e->recv = (struct engine_comm *)malloc(sizeof(struct engine_comm) * e->nr_nodes)) == NULL)
		return error(MDCERR_malloc);
	for(k = 0 ; k < e->nr_nodes ; k++) {
		if((e->send[k].cellid = (int *)malloc(sizeof(int) * 100)) == NULL)
			return error(MDCERR_malloc);
		e->send[k].size = 100;
		e->send[k].count = 0;
		if((e->recv[k].cellid = (int *)malloc(sizeof(int) * 100)) == NULL)
			return error(MDCERR_malloc);
		e->recv[k].size = 100;
		e->recv[k].count = 0;
	}

	/* Un-mark all cells. */
	for(cid = 0 ; cid < s->nr_cells ; cid++)
		s->cells[cid].flags &= ~cell_flag_marked;

	/* Loop over each cell pair... */
	for(i = 0 ; i < s->nr_tasks ; i++) {

		/* Is this task a pair? */
		if(s->tasks[i].type != task_type_pair)
			continue;

		/* Get the cells in this pair. */
		cid = s->tasks[i].i;
		cjd = s->tasks[i].j;
		ci = &(s->cells[ cid ]);
		cj = &(s->cells[ cjd ]);

		/* If it is a ghost-ghost pair, skip it. */
		if((ci->flags & cell_flag_ghost) && (cj->flags & cell_flag_ghost))
			continue;

		/* Mark the cells. */
		ci->flags |= cell_flag_marked;
		cj->flags |= cell_flag_marked;

		/* Make cj the ghost cell and bail if both are real. */
		if(ci->flags & cell_flag_ghost) {
			ct = ci; cj = ct;
			k = cid; cid = cjd; cjd = k;
		}
		else if(!(cj->flags & cell_flag_ghost))
			continue;

		/* Store the communication between cid and cjd. */
		/* Store the send, if not already there... */
		for(k = 0 ; k < e->send[cj->nodeID].count && e->send[cj->nodeID].cellid[k] != cid ; k++);
		if(k == e->send[cj->nodeID].count) {
			if(e->send[cj->nodeID].count == e->send[cj->nodeID].size) {
				e->send[cj->nodeID].size += 100;
				if((e->send[cj->nodeID].cellid = (int *)realloc(e->send[cj->nodeID].cellid, sizeof(int) * e->send[cj->nodeID].size)) == NULL)
					return error(MDCERR_malloc);
			}
			e->send[cj->nodeID].cellid[ e->send[cj->nodeID].count++ ] = cid;
		}
		/* Store the recv, if not already there... */
		for(k = 0 ; k < e->recv[cj->nodeID].count && e->recv[cj->nodeID].cellid[k] != cjd ; k++);
		if(k == e->recv[cj->nodeID].count) {
			if(e->recv[cj->nodeID].count == e->recv[cj->nodeID].size) {
				e->recv[cj->nodeID].size += 100;
				if((e->recv[cj->nodeID].cellid = (int *)realloc(e->recv[cj->nodeID].cellid, sizeof(int) * e->recv[cj->nodeID].size)) == NULL)
					return error(MDCERR_malloc);
			}
			e->recv[cj->nodeID].cellid[ e->recv[cj->nodeID].count++ ] = cjd;
		}

	}

	/* Nuke all ghost-ghost tasks. */
	i = 0;
	while(i < s->nr_tasks) {

		/* Pair? */
		if(s->tasks[i].type == task_type_pair) {

			/* Get the cells in this pair. */
			ci = &(s->cells[ s->tasks[i].i ]);
			cj = &(s->cells[ s->tasks[i].j ]);

			/* If it is a ghost-ghost pair, skip it. */
			if((ci->flags & cell_flag_ghost) && (cj->flags & cell_flag_ghost))
				s->tasks[i] = s->tasks[ --(s->nr_tasks) ];
			else
				i += 1;

		}

		/* Self? */
		else if(s->tasks[i].type == task_type_self) {

			/* Get the cells in this pair. */
			ci = &(s->cells[ s->tasks[i].i ]);

			/* If it is a ghost-ghost pair, skip it. */
			if(ci->flags & cell_flag_ghost)
				s->tasks[i] = s->tasks[ --(s->nr_tasks) ];
			else
				i += 1;

		}

		/* Sort? */
		else if(s->tasks[i].type == task_type_sort) {

			/* Get the cells in this pair. */
			ci = &(s->cells[ s->tasks[i].i ]);

			/* If it is a ghost-ghost pair, skip it. */
			if(!(ci->flags & cell_flag_marked))
				s->tasks[i] = s->tasks[ --(s->nr_tasks) ];
			else
				i += 1;

		}

	}

	/* Clear all task dependencies and re-link each sort task with its cell. */
	for(i = 0 ; i < s->nr_tasks ; i++) {
		s->tasks[i].nr_unlock = 0;
		if(s->tasks[i].type == task_type_sort) {
			s->cells[ s->tasks[i].i ].sort = &s->tasks[i];
			s->tasks[i].flags = 0;
		}
	}

	/* Run through the tasks and make each pair depend on the sorts.
       Also set the flags for each sort. */
	for(k = 0 ; k < s->nr_tasks ; k++)
		if(s->tasks[k].type == task_type_pair) {
			if(task_addunlock(s->cells[ s->tasks[k].i ].sort, &s->tasks[k]) != 0 ||
					task_addunlock(s->cells[ s->tasks[k].j ].sort, &s->tasks[k]) != 0)
				return error(MDCERR_task);
			s->cells[ s->tasks[k].i ].sort->flags |= 1 << s->tasks[k].flags;
			s->cells[ s->tasks[k].j ].sort->flags |= 1 << s->tasks[k].flags;
		}


	/* Empty unmarked cells. */
	for(k = 0 ; k < s->nr_cells ; k++)
		if(!(s->cells[k].flags & cell_flag_marked))
			space_cell_flush(&s->cells[k], s->partlist, s->celllist);

	/* Set ghost markings on particles. */
	for(cid = 0 ; cid < s->nr_cells ; cid++)
		if(s->cells[cid].flags & cell_flag_ghost)
			for(k = 0 ; k < s->cells[cid].count ; k++)
				s->cells[cid].parts[k].flags |= PARTICLE_GHOST;

	/* Fill the cid lists with marked, local and ghost cells. */
	s->nr_real = 0; s->nr_ghost = 0; s->nr_marked = 0;
	for(cid = 0 ; cid < s->nr_cells ; cid++)
		if(s->cells[cid].flags & cell_flag_marked) {
			s->cid_marked[ s->nr_marked++ ] = cid;
			if(s->cells[cid].flags & cell_flag_ghost) {
				s->cells[cid].id = -s->nr_cells;
				s->cid_ghost[ s->nr_ghost++ ] = cid;
			}
			else {
				s->cells[cid].id = s->nr_real;
				s->cid_real[ s->nr_real++ ] = cid;
			}
		}

	/* Done deal. */
	return S_OK;

}

#if defined(HAVE_CUDA)

HRESULT cuda::engine_split_gpu(struct engine *e, int N, int flags) {
	// Single GPU only
	if(N == 1) {
		for(int i = 0; i < e->s.nr_cells; i++) e->s.cells[i].GPUID = 0;
		return S_OK;
	}
	return S_OK;
}

#endif

/* Interior, recursive function that actually does the split. */
static int engine_split_bisect_rec(struct engine *e, int N_min, int N_max,
		int x_min, int x_max, int y_min, int y_max, int z_min, int z_max) {

	int i, j, k, m, Nm;
	int hx, hy, hz;
	unsigned int flag = 0;
	struct space_cell *c;

	/* Check inputs. */
	if(x_max < x_min || y_max < y_min || z_max < z_min)
		return error(MDCERR_domain);

	/* Is there nothing left to split? */
	if(N_min == N_max) {

		/* Flag as ghost or not? */
		if(N_min != e->nodeID)
			flag = cell_flag_ghost;

		/* printf("engine_split_bisect: marking range [ %i..%i, %i..%i, %i..%i ] with flag %i.\n",
            x_min, x_max, y_min, y_max, z_min, z_max, flag); */

		/* Run through the cells. */
		for(i = x_min ; i < x_max ; i++)
			for(j = y_min ; j < y_max ; j++)
				for(k = z_min ; k < z_max ; k++) {
					c = &(e->s.cells[ space_cellid(&(e->s),i,j,k) ]);
					c->flags |= flag;
					c->nodeID = N_min;
				}
	}

	/* Otherwise, bisect. */
	else {

		hx = x_max - x_min;
		hy = y_max - y_min;
		hz = z_max - z_min;
		Nm = (N_min + N_max) / 2;

		/* Is the x-axis the largest? */
		if(hx > hy && hx > hz) {
			m = (x_min + x_max) / 2;
			if(engine_split_bisect_rec(e, N_min, Nm, x_min, m, y_min, y_max, z_min, z_max) < 0 ||
					engine_split_bisect_rec(e, Nm+1, N_max, m, x_max, y_min, y_max, z_min, z_max) < 0)
				return error(MDCERR_engine);
		}

		/* Nope, maybe the y-axis? */
		else if(hy > hz) {
			m = (y_min + y_max) / 2;
			if(engine_split_bisect_rec(e, N_min, Nm, x_min, x_max, y_min, m, z_min, z_max) < 0 ||
					engine_split_bisect_rec(e, Nm+1, N_max, x_min, x_max, m, y_max, z_min, z_max) < 0)
				return error(MDCERR_engine);
		}

		/* Then it has to be the z-axis. */
		else {
			m = (z_min + z_max) / 2;
			if(engine_split_bisect_rec(e, N_min, Nm, x_min, x_max, y_min, y_max, z_min, m) < 0 ||
					engine_split_bisect_rec(e, Nm+1, N_max, x_min, x_max, y_min, y_max, m, z_max) < 0)
				return error(MDCERR_engine);
		}

	}

	/* So far, so good! */
	return S_OK;

}

HRESULT TissueForge::engine_split_bisect(struct engine *e, int N, int particle_flags) {

	/* Check inputs. */
	if(e == NULL)
		return error(MDCERR_null);

	/* Call the recursive bisection. */
	if(engine_split_bisect_rec(e, 0, N-1, 0, e->s.cdim[0], 0, e->s.cdim[1], 0, e->s.cdim[2]) < 0)
		return error(MDCERR_engine);

	/* Store the number of nodes. */
	e->nr_nodes = N;

	/* Call it a day. */
	return S_OK;

}

HRESULT TissueForge::engine_flush(struct engine *e) {

	/* check input. */
	if(e == NULL)
		return error(MDCERR_null);

	/* Clear the space. */
	if(space_flush(&e->s) != S_OK)
		return error(MDCERR_space);

	/* done for now. */
	return S_OK;

}

HRESULT TissueForge::engine_flush_ghosts(struct engine *e) {

	/* check input. */
	if(e == NULL)
		return error(MDCERR_null);

	/* Clear the space. */
	if(space_flush_ghosts(&e->s) != S_OK)
		return error(MDCERR_space);

	/* done for now. */
	return S_OK;

}

int TissueForge::engine_unload(struct engine *e, FPTYPE *x, FPTYPE *v, int *type, int *pid, int *vid, FPTYPE *q, unsigned int *flags, FPTYPE *epot, int N) {

	struct Particle *p;
	struct space_cell *c;
	int j, k, cid, count = 0, *ind;
	FPTYPE epot_acc = 0.0;

	/* check the inputs. */
	if(e == NULL) {
		error(MDCERR_null);
		return -1;
	}

	/* Allocate and fill the indices. */
	if((ind = (int *)alloca(sizeof(int) * (e->s.nr_cells + 1))) == NULL) {
		error(MDCERR_malloc);
		return -1;
	}
	ind[0] = 0;
	for(k = 0 ; k < e->s.nr_cells ; k++)
		ind[k+1] = ind[k] + e->s.cells[k].count;
	if(ind[e->s.nr_cells] > N) {
		error(MDCERR_range);
		return -1;
	}

	/* Loop over each cell. */
#pragma omp parallel for schedule(static), private(cid,count,c,k,p,j), reduction(+:epot_acc)
	for(cid = 0 ; cid < e->s.nr_cells ; cid++) {

		/* Get a hold of the cell. */
		c = &(e->s.cells[cid]);
		count = ind[cid];

		/* Collect the potential energy if requested. */
		epot_acc += c->epot;

		/* Loop over the parts in this cell. */
		for(k = 0 ; k < c->count ; k++) {

			/* Get a hold of the particle. */
			p = &(c->parts[k]);

			/* get this particle's data, where requested. */
			if(x != NULL)
				for(j = 0 ; j < 3 ; j++)
					x[count*3+j] = c->origin[j] + p->x[j];
			if(v != NULL)
				for(j = 0 ; j < 3 ; j++)
					v[count*3+j] = p->v[j];
			if(type != NULL)
				type[count] = p->typeId;
			if(pid != NULL)
				pid[count] = p->id;
			if(vid != NULL)
				vid[count] = p->vid;
			if(q != NULL)
				q[count] = p->q;
			if(flags != NULL)
				flags[count] = p->flags;

			/* Step-up the counter. */
			count += 1;

		}

	}

	/* Write back the potential energy, if requested. */
	if(epot != NULL)
		*epot += epot_acc;

	/* to the pub! */
	return ind[e->s.nr_cells];

}

int TissueForge::engine_unload_marked(struct engine *e, FPTYPE *x, FPTYPE *v, int *type, int *pid, int *vid, FPTYPE *q, unsigned int *flags, FPTYPE *epot, int N) {

	struct Particle *p;
	struct space_cell *c;
	int j, k, cid, count = 0, *ind;
	FPTYPE epot_acc = 0.0;

	/* check the inputs. */
	if(e == NULL) {
		error(MDCERR_null);
		return -1;
	}

	/* Allocate and fill the indices. */
	if((ind = (int *)alloca(sizeof(int) * (e->s.nr_cells + 1))) == NULL) {
		error(MDCERR_malloc);
		return -1;
	}
	ind[0] = 0;
	for(k = 0 ; k < e->s.nr_cells ; k++)
		if(e->s.cells[k].flags & cell_flag_marked)
			ind[k+1] = ind[k] + e->s.cells[k].count;
		else
			ind[k+1] = ind[k];
	if(ind[e->s.nr_cells] > N) {
		error(MDCERR_range);
		return -1;
	}

	/* Loop over each cell. */
#pragma omp parallel for schedule(static), private(cid,count,c,k,p,j), reduction(+:epot_acc)
	for(cid = 0 ; cid < e->s.nr_marked ; cid++) {

		/* Get a hold of the cell. */
		c = &(e->s.cells[e->s.cid_marked[cid]]);
		count = ind[e->s.cid_marked[cid]];

		/* Collect the potential energy if requested. */
		epot_acc += c->epot;

		/* Loop over the parts in this cell. */
		for(k = 0 ; k < c->count ; k++) {

			/* Get a hold of the particle. */
			p = &(c->parts[k]);

			/* get this particle's data, where requested. */
			if(x != NULL)
				for(j = 0 ; j < 3 ; j++)
					x[count*3+j] = c->origin[j] + p->x[j];
			if(v != NULL)
				for(j = 0 ; j < 3 ; j++)
					v[count*3+j] = p->v[j];
			if(type != NULL)
				type[count] = p->typeId;
			if(pid != NULL)
				pid[count] = p->id;
			if(vid != NULL)
				vid[count] = p->vid;
			if(q != NULL)
				q[count] = p->q;
			if(flags != NULL)
				flags[count] = p->flags;

			/* Step-up the counter. */
			count += 1;

		}

	}

	/* Write back the potential energy, if requested. */
	if(epot != NULL)
		*epot += epot_acc;

	/* to the pub! */
	return ind[e->s.nr_cells];

}

int TissueForge::engine_unload_strays(struct engine *e, FPTYPE *x, FPTYPE *v, int *type, int *pid, int *vid, FPTYPE *q, unsigned int *flags, FPTYPE *epot, int N) {

	struct Particle *p;
	struct space_cell *c;
	int j, k, cid, count = 0;
	FPTYPE epot_acc = 0.0;

	/* check the inputs. */
	if(e == NULL) {
		error(MDCERR_null);
		return -1;
	}

	/* Loop over each cell. */
	for(cid = 0 ; cid < e->s.nr_real ; cid++) {

		/* Get a hold of the cell. */
		c = &(e->s.cells[e->s.cid_real[cid]]);

		/* Collect the potential energy if requested. */
		epot_acc += c->epot;

		/* Loop over the parts in this cell. */
		for(k = c->count-1 ; k >= 0 && !(c->parts[k].flags & PARTICLE_GHOST) ; k--) {

			/* Get a hold of the particle. */
			p = &(c->parts[k]);
			if(p->flags & PARTICLE_GHOST)
				continue;

			/* get this particle's data, where requested. */
			if(x != NULL)
				for(j = 0 ; j < 3 ; j++)
					x[count*3+j] = c->origin[j] + p->x[j];
			if(v != NULL)
				for(j = 0 ; j < 3 ; j++)
					v[count*3+j] = p->v[j];
			if(type != NULL)
				type[count] = p->typeId;
			if(pid != NULL)
				pid[count] = p->id;
			if(vid != NULL)
				vid[count] = p->vid;
			if(q != NULL)
				q[count] = p->q;
			if(flags != NULL)
				flags[count] = p->flags;

			/* increase the counter. */
			count += 1;

		}

	}

	/* Write back the potential energy, if requested. */
	if(epot != NULL)
		*epot += epot_acc;

	/* to the pub! */
	return count;

}

HRESULT TissueForge::engine_load(struct engine *e, FPTYPE *x, FPTYPE *v, int *type, int *pid, int *vid, FPTYPE *q, unsigned int *flags, int N) {

    struct Particle p = {};
	int j, k;

	/* check the inputs. */
	if(e == NULL || x == NULL || type == NULL)
		return error(MDCERR_null);

	/* init the velocity and charge in case not specified. */
	p.v[0] = 0.0; p.v[1] = 0.0; p.v[2] = 0.0;
	p.f[0] = 0.0; p.f[1] = 0.0; p.f[2] = 0.0;
	p.q = 0.0;
	p.flags = PARTICLE_NONE;

	/* loop over the entries. */
	for(j = 0 ; j < N ; j++) {

		/* set the particle data. */
		p.typeId = type[j];
		if(pid != NULL)
			p.id = pid[j];
		else
			p.id = j;
		if(vid != NULL)
			p.vid = vid[j];
		if(flags != NULL)
			p.flags = flags[j];
		if(v != NULL)
			for(k = 0 ; k < 3 ; k++)
				p.v[k] = v[j*3+k];
		if(q != 0)
			p.q = q[j];

		/* add the part to the space. */
		if(engine_addpart(e, &p, &x[3*j], NULL) != S_OK)
			return error(MDCERR_space);

	}

	/* to the pub! */
	return S_OK;

}

HRESULT TissueForge::engine_load_ghosts(struct engine *e, FPTYPE *x, FPTYPE *v, int *type, int *pid, int *vid, FPTYPE *q, unsigned int *flags, int N) {

    struct Particle p = {};
	struct space *s;
	int j, k;

	/* check the inputs. */
	if(e == NULL || x == NULL || type == NULL)
		return error(MDCERR_null);

	/* Get a handle on the space. */
	s = &(e->s);

	/* init the velocity and charge in case not specified. */
	p.v[0] = 0.0; p.v[1] = 0.0; p.v[2] = 0.0;
	p.f[0] = 0.0; p.f[1] = 0.0; p.f[2] = 0.0;
	p.q = 0.0;
	p.flags = PARTICLE_GHOST;

	/* loop over the entries. */
	for(j = 0 ; j < N ; j++) {

		/* set the particle data. */
		p.typeId = type[j];
		if(pid != NULL)
			p.id = pid[j];
		else
			p.id = j;
		if(vid != NULL)
			p.vid = vid[j];
		if(flags != NULL)
			p.flags = flags[j] | PARTICLE_GHOST;
		if(v != NULL)
			for(k = 0 ; k < 3 ; k++)
				p.v[k] = v[j*3+k];
		if(q != 0)
			p.q = q[j];

		/* add the part to the space. */
		if(engine_addpart(e, &p, &x[3*j], NULL) != S_OK)
			return error(MDCERR_space);

	}

	/* to the pub! */
	return S_OK;

}

int TissueForge::engine_gettype(struct engine *e, char *name) {

	int k;

	/* check for nonsense. */
	if(e == NULL || name == NULL) {
		error(MDCERR_null);
		return -1;
	}

	/* Loop over the types... */
	for(k = 0 ; k < e->nr_types ; k++) {

		/* Compare the name. */
		if(strcmp(e->types[k].name, name) == 0)
			return k;

	}

	/* Otherwise, nothing found... */
	error(MDCERR_range);
	return -1;

}

int TissueForge::engine_gettype2(struct engine *e, char *name2) {

	int k;

	/* check for nonsense. */
	if(e == NULL || name2 == NULL) {
		error(MDCERR_null);
		return -1;
	}

	/* Loop over the types... */
	for(k = 0 ; k < e->nr_types ; k++) {

		/* Compare the name. */
		if(strcmp(e->types[k].name2, name2) == 0)
			return k;

	}

	/* Otherwise, nothing found... */
	error(MDCERR_range);
	return -1;

}

int TissueForge::engine_addtype(struct engine *e, FPTYPE mass, FPTYPE charge,
        const char *name, const char *name2) {

    /* check for nonsense. */
    if(e == NULL) {
		error(MDCERR_null);
		return -1;
	}
    if(e->nr_types >= e->max_type) {
		error(MDCERR_range);
		return -1;
	}

    ParticleType *type = ParticleType_ForEngine(e, mass, charge, name, name2);
    return type != NULL ? type->id : -1;
}

HRESULT TissueForge::engine_addpot(struct engine *e, struct Potential *p, int i, int j) {
	TF_Log(LOG_DEBUG);

	/* check for nonsense. */
	if(e == NULL)
		return error(MDCERR_null);
	if(i < 0 || i >= e->nr_types || j < 0 || j >= e->nr_types)
		return error(MDCERR_range);

    Potential **pots = p->flags & POTENTIAL_BOUND ? e->p_cluster : e->p;

	/* store the potential. */
	pots[ i * e->max_type + j ] = p;

    if(i != j) pots[ j * e->max_type + i ] = p;

	#if defined(HAVE_CUDA)
	if(e->flags & engine_flag_cuda && cuda::engine_cuda_refresh_pots(e) != S_OK)
		return error(MDCERR_cuda);
	#endif

	/* end on a good note. */
	return S_OK;
}

HRESULT TissueForge::engine_addfluxes(struct engine *e, struct Fluxes *f, int i, int j) {
	TF_Log(LOG_DEBUG);

	/* check for nonsense. */
	if(e == NULL)
		return error(MDCERR_null);
	if(i < 0 || i >= e->nr_types || j < 0 || j >= e->nr_types)
		return error(MDCERR_range);

	Fluxes **fluxes = e->fluxes;
	fluxes[i * e->max_type + j] = f;

	if(i != j) fluxes[j * e->max_type + i] = f;

	#if defined(HAVE_CUDA)
	if(e->flags & engine_flag_cuda && cuda::engine_cuda_refresh_fluxes(e) != S_OK)
		return error(MDCERR_cuda);
	#endif

	return S_OK;
}

Fluxes *TissueForge::engine_getfluxes(struct engine *e, int i, int j) {
	TF_Log(LOG_DEBUG);

	/* check for nonsense. */
	if(e == NULL) {
		error(MDCERR_null);
		return NULL;
	}
	if(i < 0 || i >= e->nr_types || j < 0 || j >= e->nr_types) {
		error(MDCERR_range);
		return NULL;
	}

	Fluxes **fluxes = e->fluxes;
	return fluxes[i * e->max_type + j];
}

HRESULT TissueForge::engine_add_singlebody_force(struct engine *e, struct Force *p, int i) {
    /* check for nonsense. */
    if(e == NULL)
        return error(MDCERR_null);
    if(i < 0 || i >= e->nr_types)
        return error(MDCERR_range);

    /* store the force. */
    e->forces[i] = p;

    if(p->isCustom()) e->custom_forces.push_back((CustomForce*)p);

    /* end on a good note. */
    return S_OK;
}

#if defined(HAVE_CUDA)

HRESULT cuda::engine_toCUDA(struct engine *e) {
	
	// Check input

	if(e == NULL)
		return error(MDCERR_null);

	// Check state

	if(!(e->flags & engine_flag_initialized)) 
		return error(MDCERR_cuda);

	// If already on cuda, do nothing

	if(e->flags & engine_flag_cuda)
		return S_OK;

	// Start cuda run mode

	if(engine_cuda_load(e) != S_OK)
		return error(MDCERR_cuda);

	e->flags |= engine_flag_cuda;

	return S_OK;

}

HRESULT cuda::engine_fromCUDA(struct engine *e) {
	
	// Check input

	if(e == NULL)
		return error(MDCERR_null);

	// Check state

	if(!(e->flags & engine_flag_initialized)) 
		return error(MDCERR_cuda);

	// If not on cuda, do nothing

	if(!(e->flags & engine_flag_cuda))
		return S_OK;

	// Shut down cuda run mode

	if(engine_cuda_finalize(e) != S_OK)
		return S_OK;

	e->flags &= ~engine_flag_cuda;

	// Prep restarting local run mode

    /* Run through the tasks and reset the waits. */
    for (int k = 0 ; k < e->s.nr_tasks ; k++)
        for (int j = 0 ; j < e->s.tasks[k].nr_unlock ; j++) 
			e->s.tasks[k].unlock[j]->wait = 0;

	return S_OK;

}

#endif

HRESULT TissueForge::engine_start(struct engine *e, int nr_runners, int nr_queues) {

	int cid, pid, k, i;
	struct space_cell *c;
	struct Particle *p;
	struct space *s = &e->s;

	/* Is MPI really needed? */
	if(e->flags & engine_flag_mpi && e->nr_nodes == 1)
		e->flags &= ~(engine_flag_mpi | engine_flag_async);

#ifdef WITH_MPI
	/* Set up async communication? */
	if(e->particle_flags & engine_flag_async) {

		/* Init the mutex and condition variable for the asynchronous communication. */
		if(pthread_mutex_init(&e->xchg_mutex, NULL) != 0 ||
				pthread_cond_init(&e->xchg_cond, NULL) != 0 ||
				pthread_mutex_init(&e->xchg2_mutex, NULL) != 0 ||
				pthread_cond_init(&e->xchg2_cond, NULL) != 0)
			return error(MDCERR_pthread);

		/* Set the exchange flags. */
		e->xchg_started = 0;
		e->xchg_running = 0;
		e->xchg2_started = 0;
		e->xchg2_running = 0;

		/* Start a thread with the async exchange. */
		if(pthread_create(&e->thread_exchg, NULL, (void *(*)(void *))engine_exchange_async_run, e) != 0)
			return error(MDCERR_pthread);
		if(pthread_create(&e->thread_exchg2, NULL, (void *(*)(void *))engine_exchange_rigid_async_run, e) != 0)
			return error(MDCERR_pthread);

	}
#endif

	/* Fill-in the Verlet lists if needed. */
	if(e->flags & engine_flag_verlet) {

		/* Shuffle the domain. */
		if(engine_shuffle(e) != S_OK)
			return error(MDCERR_engine);

		/* Store the current positions as a reference. */
#pragma omp parallel for schedule(static), private(cid,c,pid,p,k)
		for(cid = 0 ; cid < s->nr_real ; cid++) {
			c = &(s->cells[s->cid_real[cid]]);
			if(c->oldx == NULL || c->oldx_size < c->count) {
				free(c->oldx);
				c->oldx_size = c->size + 20;
				c->oldx = (FPTYPE *)malloc(sizeof(FPTYPE) * 4 * c->oldx_size);
			}
			for(pid = 0 ; pid < c->count ; pid++) {
				p = &(c->parts[pid]);
				for(k = 0 ; k < 3 ; k++)
					c->oldx[ 4*pid + k ] = p->x[k];
			}
		}

		/* Re-set the Verlet rebuild flag. */
		s->verlet_rebuild = 1;

	}

	/* Is MPI really needed? */
	if(e->flags & engine_flag_mpi && e->nr_nodes == 1)
		e->flags &= ~engine_flag_mpi;

	/* Do we even need runners? */
	if(e->flags & engine_flag_cuda) {

		/* Set the number of runners. */
		e->nr_runners = nr_runners;

#if defined(HAVE_CUDA)
		/* Load the potentials and pairs to the CUDA device. */
		if(cuda::engine_cuda_load(e) != S_OK)
			return error(MDCERR_cuda);
#else
		/* Was not compiled with CUDA support. */
		return error(MDCERR_nocuda);
#endif

	}
	else {

		/* Allocate the queues */
		if((e->queues = (struct queue *)malloc(sizeof(struct queue) * nr_queues)) == NULL)
			return error(MDCERR_malloc);
		e->nr_queues = nr_queues;

		/* Initialize  and fill the queues. */
		for(i = 0 ; i < e->nr_queues ; i++)
			if(queue_init(&e->queues[i], 2*s->nr_tasks/e->nr_queues, s, s->tasks) != S_OK)
				return error(MDCERR_queue);
		for(i = 0 ; i < s->nr_tasks ; i++)
			if(queue_insert(&e->queues[ i % e->nr_queues ], &s->tasks[i]) < 0)
				return error(MDCERR_queue);

		/* (Allocate the runners */
				if((e->runners = (struct runner *)malloc(sizeof(struct runner) * nr_runners)) == NULL)
					return error(MDCERR_malloc);
				e->nr_runners = nr_runners;

				/* initialize the runners. */
				for(i = 0 ; i < nr_runners ; i++)
					if(runner_init(&e->runners[ i ], e, i) != S_OK)
						return error(MDCERR_runner);

				/* wait for the runners to be in place */
				while (e->barrier_count != e->nr_runners)
					if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
						return error(MDCERR_pthread);

	}

	/* Set the number of runners. */
	e->nr_runners = nr_runners;

	/* all is well... */
	return S_OK;
}

HRESULT TissueForge::engine_nonbond_eval(struct engine *e) {

	TF_Log(LOG_TRACE);

	int k;

	/* Re-set the queues. */
	for(k = 0 ; k < e->nr_queues ; k++)
		e->queues[k].next = 0;

	/* open the door for the runners */
	e->barrier_count = -e->barrier_count;
	if(e->nr_runners == 1) {
		if (pthread_cond_signal(&e->barrier_cond) != 0)
			return error(MDCERR_pthread);
	}
	else {
		if (pthread_cond_broadcast(&e->barrier_cond) != 0)
			return error(MDCERR_pthread);
	}

	/* wait for the runners to come home */
	while (e->barrier_count < e->nr_runners)
		if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
			return error(MDCERR_pthread);

	TF_Log(LOG_TRACE);

	/* All in a days work. */
	return S_OK;

}

HRESULT TissueForge::engine_fluxonly_eval(struct engine* e) {

	TF_Log(LOG_TRACE);

	e->integrator_flags |= INTEGRATOR_FLUX_SUBSTEP;

	if(engine_nonbond_eval(e) != S_OK) {
		e->integrator_flags &= ~INTEGRATOR_FLUX_SUBSTEP;
		return error(MDCERR_engine);
	}

	e->integrator_flags &= ~INTEGRATOR_FLUX_SUBSTEP;

	/** Reset tasks for future calls in this step */
    if(space_prepare_tasks(&e->s) != S_OK) 
		return error(MDCERR_engine);

	return S_OK;
}

HRESULT TissueForge::engine_step(struct engine *e) {

	TF_Log(LOG_TRACE);

	int i;
    util::WallTime wt;
    util::PerformanceTimer t(engine_timer_step);
    update_steps_per_second();
    
	/* increase the time stepper */
	e->time += 1;

	if(engine_force_prep(e) != S_OK) 
		return error(MDCERR_engine);

	// Pre-step subengines
	for(auto &se : e->subengines) 
		if((i = se->preStepStart()) != S_OK) 
			return error(MDCERR_subengine);
	for(auto &se : e->subengines) 
		if((i = se->preStepJoin()) != S_OK) 
			return error(MDCERR_subengine);

	if(engine_advance(e) != S_OK) 
		return error(MDCERR_engine);

    /* Shake the particle positions? */
    if(e->nr_rigids > 0) {
        util::PerformanceTimer tr(engine_timer_rigid);

		/* Resolve the constraints. */
		if(engine_rigid_eval(e) != 0)
			return error(MDCERR_engine);
	}

    for(CustomForce* p : e->custom_forces) {
        p->onTime(e->time * e->dt);
    }

	// Post-step subengines
	for(auto &se : e->subengines) 
		if((i = se->postStepStart()) != S_OK) 
			return error(MDCERR_subengine);
	for(auto &se : e->subengines) 
		if((i = se->postStepJoin()) != S_OK) 
			return error(MDCERR_subengine);

	TF_Log(LOG_TRACE);

	/* return quietly */
	return S_OK;
}

HRESULT TissueForge::engine_force_prep(struct engine *e) {

	TF_Log(LOG_TRACE);

	e->step_flux = 0;

    ticks tic = getticks();
	
    // clear the energy on the types
    // TODO: should go in prepare space for better performance
    engine_kinetic_energy(e);
	e->timers[engine_timer_kinetic] += getticks() - tic;

    /* prepare the space, sets forces to zero */
    tic = getticks();
    if(space_prepare(&e->s) != S_OK)
        return error(MDCERR_space);
    e->timers[engine_timer_prepare] += getticks() - tic;

    /* Make sure the verlet lists are up to date. */
    if(e->flags & engine_flag_verlet) {

        /* Start the clock. */
        tic = getticks();

        /* Check particle movement and update cells if necessary. */
        if(engine_verlet_update(e) != S_OK) {
            return error(MDCERR_engine);
        }

        /* Store the timing. */
        e->timers[engine_timer_verlet] += getticks() - tic;
    }


    /* Otherwise, if async MPI, move the particles accross the
       node boundaries. */
    else { // if(e->flags & engine_flag_async) {
        tic = getticks();
        if(engine_shuffle(e) != S_OK) {
            return error(MDCERR_space);
        }
        e->timers[engine_timer_shuffle] += getticks() - tic;
    }


#ifdef WITH_MPI
    /* Re-distribute the particles to the processors. */
    if(e->particle_flags & engine_flag_mpi) {

        /* Start the clock. */
        tic = getticks();

        if(e->particle_flags & engine_flag_async) {
            if(engine_exchange_async(e) < 0)
                return error(MDCERR_mpi);
        }
        else {
            if(engine_exchange(e) < 0)
                return error(MDCERR_mpi);
        }

        /* Store the timing. */
        e->timers[engine_timer_exchange1] += getticks() - tic;

    }
#endif

	TF_Log(LOG_TRACE);

    return S_OK;
}

HRESULT TissueForge::engine_force(struct engine *e) {

	TF_Log(LOG_TRACE);

    ticks tic = getticks();

    /* Compute the non-bonded interactions. */
    tic = getticks();
    #if defined(HAVE_CUDA)
        if(e->flags & engine_flag_cuda) {
            if(cuda::engine_nonbond_cuda(e) != S_OK)
                return error(MDCERR_cuda);
			
			space *s = &e->s;
			Force **forces = e->forces;
			auto func_eval_forces = [&s, &forces] (int cid) -> void {
				space_cell *c = &s->cells[s->cid_real[cid]];
				for(int pid = 0; pid < c->count; pid++) {
					Particle *p = &c->parts[pid];
					Force *force = forces[p->typeId];
					if(force) 
						force->func(force, p, p->f);
				}
			};
			parallel_for(s->nr_real, func_eval_forces);
            }
        else
    #endif
    if(engine_nonbond_eval(e) != S_OK) {
        return error(MDCERR_engine);
    }

    e->timers[engine_timer_nonbond] += getticks() - tic;

    /* Clear the verlet-rebuild flag if it was set. */
    if(e->flags & engine_flag_verlet && e->s.verlet_rebuild)
        e->s.verlet_rebuild = 0;

    /* Do bonded interactions. */
    tic = getticks();
    if(e->flags & engine_flag_sets) {
        if(engine_bonded_eval_sets(e) != S_OK)
            return error(MDCERR_engine);
    }
    else {
        if(engine_bonded_eval(e) != S_OK)
            return error(MDCERR_engine);
    }
    e->timers[engine_timer_bonded] += getticks() - tic;

	TF_Log(LOG_TRACE);

    return S_OK;
}

HRESULT TissueForge::engine_barrier(struct engine *e) {

	/* lock the barrier mutex */
	if (pthread_mutex_lock(&e->barrier_mutex) != 0)
		return error(MDCERR_pthread);

	/* wait for the barrier to close */
	while (e->barrier_count < 0)
		if (pthread_cond_wait(&e->barrier_cond,&e->barrier_mutex) != 0)
			return error(MDCERR_pthread);

	/* if i'm the last thread in, signal that the barrier is full */
	if (++e->barrier_count == e->nr_runners) {
		if (pthread_cond_signal(&e->done_cond) != 0)
			return error(MDCERR_pthread);
	}

	/* wait for the barrier to re-open */
	while (e->barrier_count > 0)
		if (pthread_cond_wait(&e->barrier_cond,&e->barrier_mutex) != 0)
			return error(MDCERR_pthread);

	/* if i'm the last thread out, signal to those waiting to get back in */
	if (++e->barrier_count == 0)
		if (pthread_cond_broadcast(&e->barrier_cond) != 0)
			return error(MDCERR_pthread);

	/* free the barrier mutex */
	if (pthread_mutex_unlock(&e->barrier_mutex) != 0)
		return error(MDCERR_pthread);

	/* all is well... */
	return S_OK;

}

#ifdef WITH_MPI
int engine_init_mpi(struct engine *e, const FPTYPE *origin, const FPTYPE *dim, FPTYPE *L, FPTYPE cutoff, unsigned int period, int max_type, unsigned int particle_flags, MPI_Comm comm, int rank) {

	/* Init the engine. */
	if(engine_init(e, origin, dim, L, cutoff, period, max_type, particle_flags | engine_flag_mpi) != S_OK)
		return error(MDCERR_engine);

	/* Store the MPI Comm and rank. */
	e->comm = comm;
	e->nodeID = rank;

	/* Bail. */
	return S_OK;

}
#endif

HRESULT TissueForge::engine_finalize(struct engine *e) {

    int j, k;

    /* make sure the inputs are ok */
    if(e == NULL)
        return error(MDCERR_null);

    // If running on CUDA, bring run mode back to local
	if(e->flags & engine_flag_cuda)
	#if defined(HAVE_CUDA)
		if(cuda::engine_fromCUDA(e) != S_OK)
			return error(MDCERR_cuda);
	#endif

	// Finalize subengines
	for(auto &se : e->subengines) 
		if((j = se->finalize()) != S_OK) 
			return error(MDCERR_subengine);

    /* Shut down the runners, if they were started. */
    if(e->runners != NULL) {
        for(k = 0 ; k < e->nr_runners ; k++)
            if(pthread_cancel(e->runners[k].thread) != 0)
                return error(MDCERR_pthread);
        free(e->runners);
        free(e->queues);
    }

    /* Finalize the space. */
    // if(space_finalize(&e->s) < 0)
    //     return error(MDCERR_space);

    /* Free-up the types. */
    free(e->types);

    /* Free the potentials. */
    if(e->p != NULL) {
        for(j = 0 ; j < e->nr_types ; j++) {
            for(k = j ; k < e->nr_types ; k++) {
                if(e->p[ j*e->max_type + k ] != NULL)
                    potential_clear(e->p[ j*e->max_type + k ]);
            }
        }

        for(j = 0 ; j < e->nr_types ; j++) {
            for(k = j ; k < e->nr_types ; k++) {
                if(e->p[ j*e->max_type + k ] != NULL)
                    potential_clear(e->p_cluster[ j*e->max_type + k ]);
            }
        }

        free(e->p);
    }

    /* Free the communicators, if needed. */
    if(e->flags & engine_flag_mpi) {
        for(k = 0 ; k < e->nr_nodes ; k++) {
            free(e->send[k].cellid);
            free(e->recv[k].cellid);
        }
        free(e->send);
        free(e->recv);
    }

    /* Free the bonded interactions. */
    free(e->bonds);
    free(e->angles);
    free(e->dihedrals);
    free(e->exclusions);
    free(e->rigids);
    free(e->part2rigid);

    /* If we have bonded sets, kill them. */
    for(k = 0 ; k < e->nr_sets ; k++) {
        free(e->sets[k].bonds);
        free(e->sets[k].angles);
        free(e->sets[k].dihedrals);
        free(e->sets[k].exclusions);
        free(e->sets[k].confl);
    }

    /* Clear all the counts and what not. */
    bzero(e, sizeof(struct engine));

    /* Happy and I know it... */
    return S_OK;

}

HRESULT TissueForge::engine_init(struct engine *e, const FPTYPE *origin, const FPTYPE *dim, int *cells,
        FPTYPE cutoff, BoundaryConditionsArgsContainer *boundaryConditions, int max_type, unsigned int flags, unsigned int nr_fluxsteps) {

    int cid;
    
    // TODO: total hack
    init_types = engine::nr_types;

    /* make sure the inputs are ok */
    if(e == NULL || origin == NULL || dim == NULL || cells == NULL || nr_fluxsteps < 1) {
		if(!e) 		{ TF_Log(LOG_CRITICAL) << "no engine"; }
		if(!origin) { TF_Log(LOG_CRITICAL) << "no origin"; }
		if(!dim) 	{ TF_Log(LOG_CRITICAL) << "no dim"; }
		if(!cells) 	{ TF_Log(LOG_CRITICAL) << "no cells"; }
		if(nr_fluxsteps < 1) { TF_Log(LOG_CRITICAL) << "no flux steps"; }

        return error(MDCERR_null);
	}

    // set up boundary conditions, adjust cell count if needed
	e->boundary_conditions = *boundaryConditions->create(cells);

    // figure out spatials size...
    FVector3 domain_dim {dim[0] - origin[0], dim[1] - origin[1], dim[2] - origin[2]};

    FVector3 L = {domain_dim[0] / cells[0], domain_dim[1] / cells[1], domain_dim[2] / cells[2]};

    // initialize the engine
    TF_Log(LOG_INFORMATION) << "engine: initializing the engine... ";
    TF_Log(LOG_INFORMATION) << "engine: requesting origin = [" << origin[0] << ", " << origin[1] << ", " << origin[2] << "]";
    TF_Log(LOG_INFORMATION) << "engine: requesting dimensions = [" << dim[0] << ", " << dim[1] << ", " << dim[2]  << "]";
    TF_Log(LOG_INFORMATION) << "engine: requesting cell size = [" << L[0] << ", " << L[1] << ", " << L[2] << "]";
    TF_Log(LOG_INFORMATION) << "engine: requesting cutoff = " << cutoff;
	TF_Log(LOG_INFORMATION) << "engine: requesting flux steps = " << nr_fluxsteps;

    /* default Boltzmann constant to 1 */
    e->K = 1.0;

    e->integrator_flags = 0;

	e->nr_fluxsteps = nr_fluxsteps;

    /* init the space with the given parameters */
    if(space_init(&(e->s), origin, dim, L.data(), cutoff, &e->boundary_conditions) != S_OK)
        return error(MDCERR_space);

    /* Set some flag implications. */
    if(flags & engine_flag_verlet_pseudo)
        flags |= engine_flag_verlet_pairwise;
    if(flags & engine_flag_verlet_pairwise)
        flags |= engine_flag_verlet;
    if(flags & engine_flag_cuda)
        flags |= engine_flag_nullpart;

    /* Set the flags. */
    e->flags = flags;

    /* By default there is only one node. */
    e->nr_nodes = 1;

    /* Init the timers. */
    if(engine_timers_reset(e) != S_OK)
        return error(MDCERR_engine);

    /* Init the runners to 0. */
    e->runners = NULL;
    e->nr_runners = 0;

    /* Start with no queues. */
    e->queues = NULL;
    e->nr_queues = 0;

    /* Init the bonds array. */
    e->bonds_size = 100;
    if((e->bonds = (struct Bond *)malloc(sizeof(struct Bond) * e->bonds_size)) == NULL)
        return error(MDCERR_malloc);
    e->nr_bonds = 0;
    e->nr_active_bonds = 0;

    /* Init the exclusions array. */
    e->exclusions_size = 100;
    if((e->exclusions = (struct exclusion *)malloc(sizeof(struct exclusion) * e->exclusions_size)) == NULL)
        return error(MDCERR_malloc);
    e->nr_exclusions = 0;

    /* Init the rigids array. */
    e->rigids_size = 100;
    if((e->rigids = (struct rigid *)malloc(sizeof(struct rigid) * e->rigids_size)) == NULL)
        return error(MDCERR_malloc);
    e->nr_rigids = 0;
    e->tol_rigid = 1e-6;
    e->nr_constr = 0;
    e->part2rigid = NULL;

    /* Init the angles array. */
    e->angles_size = 100;
    if((e->angles = (struct Angle *)malloc(sizeof(struct Angle) * e->angles_size)) == NULL)
        return error(MDCERR_malloc);
    e->nr_angles = 0;
	e->nr_active_angles = 0;

    /* Init the dihedrals array.		 */
    e->dihedrals_size = 100;
    if((e->dihedrals = (struct Dihedral *)malloc(sizeof(struct Dihedral) * e->dihedrals_size)) == NULL)
        return error(MDCERR_malloc);
    e->nr_dihedrals = 0;
	e->nr_active_dihedrals = 0;


    /* Init the sets. */
    e->sets = NULL;
    e->nr_sets = 0;

    /* allocate the interaction matrices */
    if((e->p = (struct Potential **)malloc(sizeof(Potential*) * e->max_type * e->max_type)) == NULL)
        return error(MDCERR_malloc);

    /* allocate the flux interaction matrices */
    if((e->fluxes =(Fluxes **)malloc(sizeof(Fluxes*) * e->max_type * e->max_type)) == NULL)
        return error(MDCERR_malloc);

    if((e->p_cluster = (struct Potential **)malloc(sizeof(Potential*) * e->max_type * e->max_type)) == NULL)
            return error(MDCERR_malloc);

    bzero(e->p, sizeof(struct Potential *) * e->max_type * e->max_type);

    bzero(e->fluxes, sizeof(struct Fluxes *) * e->max_type * e->max_type);

    bzero(e->p_cluster, sizeof(struct Potential *) * e->max_type * e->max_type);

    // init singlebody forces
    if((e->forces = (Force**)malloc(sizeof(struct Force*) * e->max_type)) == NULL)
            return error(MDCERR_malloc);
    bzero(e->forces, sizeof(struct Force*) * e->max_type);

    /* Make sortlists? */
    if(flags & engine_flag_verlet_pseudo) {
        for(cid = 0 ; cid < e->s.nr_cells ; cid++)
            if(e->s.cells[cid].flags & cell_flag_marked)
                if((e->s.cells[cid].sortlist = (unsigned int *)malloc(sizeof(unsigned int) * 13 * e->s.cells[cid].size)) == NULL)
                    return error(MDCERR_malloc);
    }

    /* init the barrier variables */
    e->barrier_count = 0;
    if(pthread_mutex_init(&e->barrier_mutex, NULL) != 0 ||
            pthread_cond_init(&e->barrier_cond, NULL) != 0 ||
            pthread_cond_init(&e->done_cond, NULL) != 0)
        return error(MDCERR_pthread);

    /* init the barrier */
    if (pthread_mutex_lock(&e->barrier_mutex) != 0)
        return error(MDCERR_pthread);
    e->barrier_count = 0;

    /* Init the comm arrays. */
    e->send = NULL;
    e->recv = NULL;

    e->integrator = EngineIntegrator::FORWARD_EULER;

    e->flags |= engine_flag_initialized;

    e->particle_max_dist_fraction = 0.05;
    
    e->_init_boundary_conditions = boundaryConditions;
    e->_init_cells[0] = cells[0];
    e->_init_cells[1] = cells[1];
    e->_init_cells[2] = cells[2];
    
    /* all is well... */
    return S_OK;
}




void TissueForge::engine_dump() {
    for(int cid = 0; cid < _Engine.s.nr_cells; ++cid) {
        space_cell *cell = &_Engine.s.cells[cid];
        for(int pid = 0; pid < cell->count; ++pid) {
            Particle *p = &cell->parts[pid];

            TF_Log(LOG_NOTICE) << "i: " << pid << ", pid: " << p->id <<
                    ", {" << p->x[0] << ", " << p->x[1] << ", " << p->x[2] << "}"
                    ", {" << p->v[0] << ", " << p->v[1] << ", " << p->v[2] << "}";

        }
    }
}

FPTYPE TissueForge::engine_kinetic_energy(struct engine *e)
{
    FPTYPE total = 0;
    
    // clear the ke in the types,
    for(int i = 0; i < engine::nr_types; ++i) {
        engine::types[i].kinetic_energy = 0;
    }
	
	std::vector<FPTYPE> worker_total(_Engine.s.nr_cells);
	std::vector<std::vector<FPTYPE> > worker_type_total(_Engine.s.nr_cells);
	auto nr_types = _Engine.nr_types;
	auto func_cell_kinetic_energy = [&, nr_types](int cid) {
		worker_total[cid] = 0;
		worker_type_total[cid] = std::vector<FPTYPE>(nr_types, 0);
		space_cell *cell = &_Engine.s.cells[cid];
        for(int pid = 0; pid < cell->count; ++pid) {
            Particle *p = &cell->parts[pid];
            ParticleType *type = &engine::types[p->typeId];
            FPTYPE e = 0.5 * type->mass * (p->v[0] * p->v[0] + p->v[1] * p->v[1] + p->v[2] * p->v[2]);
			worker_type_total[cid][type->id] += e;
            worker_total[cid] += e;
        }
	};
	parallel_for(_Engine.s.nr_cells, func_cell_kinetic_energy);

	for(int cid = 0; cid < _Engine.s.nr_cells; cid++) {
		total += worker_total[cid];
		for(int tid = 0; tid < nr_types; tid++) {
			engine::types[tid].kinetic_energy += worker_type_total[cid][tid];
		}
	}

    return total;
}

FPTYPE TissueForge::engine_temperature(struct engine *e)
{
    return e->temperature;
}

HRESULT TissueForge::engine_addpart(struct engine *e, struct Particle *p, FPTYPE *x,
        struct Particle **result)
{
    if(p->typeId < 0 || p->typeId >= e->nr_types) {
        return error(MDCERR_range);
    }

    if(space_addpart (&(e->s), p, x, result) != 0) {
        return error(MDCERR_space);
    }

    e->types[p->typeId].addpart(p->id);

    return S_OK;
}

HRESULT TissueForge::engine_addparts(struct engine *e, int nr_parts, struct Particle **parts, FPTYPE **x)
{
	int num_workers = TissueForge::ThreadPool::size();
	
	// Check input and count particles of each type

	int nr_types = e->nr_types;
	std::vector<bool> worker_check(num_workers, false);
	std::vector<std::vector<int> >worker_type_counts(num_workers);

	auto func_check = [num_workers, nr_types, nr_parts, &worker_check, &worker_type_counts, &parts](int wid) -> void { 
		worker_type_counts[wid] = std::vector<int>(nr_types, 0);
		for(int i = wid; i < nr_parts; i += num_workers) {
			if(parts[i]->typeId < 0 || parts[i]->typeId >= nr_types) {
				worker_check[wid] = true;
				return;
			}
			worker_type_counts[wid][parts[i]->typeId]++;
		}
	};
	parallel_for(num_workers, func_check);
	for(int i = 0; i < worker_check.size(); i++) 
		if(worker_check[i]) 
			return error(MDCERR_range);
	
	// Gather type counts
	std::vector<int> type_counts(nr_types, 0);
	for(int i = 0; i < nr_types; i++) 
		for(int j = 0; j < num_workers; j++) 
			type_counts[i] += worker_type_counts[j][i];

    // Add parts
	if(space_addparts (&(e->s), nr_parts, parts, x) != 0) {
        return error(MDCERR_space);
    }

	// Gather ids for type containers
    std::vector<ParticleList> ptype_lists(e->nr_types);
	for(int i = 0; i < nr_types; i++) 
		ptype_lists[i].reserve(type_counts[i]);
	for(int i = 0; i < nr_parts; i++) 
		ptype_lists[parts[i]->typeId].insert(parts[i]->id);
	
	// Add ids to type containers
	auto ptypes = e->types;
	auto func_extend = [num_workers, &ptypes, &ptype_lists](int wid) -> void {
		for(int i = wid; i < ptype_lists.size(); i += num_workers) 
			if(ptype_lists[i].nr_parts > 0) 
				ptypes[i].parts.extend(ptype_lists[i]);
	};
	parallel_for(num_workers, func_extend);

    return S_OK;
}

struct ParticleType* TissueForge::engine_type(int id)
{
    if(id >= 0 && id < engine::nr_types) {
        return &engine::types[id];
    }
    return NULL;
}

int TissueForge::engine_next_partid(struct engine *e)
{
	if(e->pids_avail.empty()) 
		return e->s.nr_parts;

	std::set<unsigned int>::iterator itr = e->pids_avail.begin();
	unsigned int pid = *itr;
	e->pids_avail.erase(itr);

	return pid;
}

HRESULT TissueForge::engine_next_partids(struct engine *e, int nr_ids, int *ids) { 
	int j = e->pids_avail.size();
	for(int i = 0; i < nr_ids; i++) {
		if(!e->pids_avail.empty()) {
			std::set<unsigned int>::iterator itr = e->pids_avail.begin();
			ids[i] = *itr;
			e->pids_avail.erase(itr);
		} 
		else {
			ids[i] = e->s.nr_parts + j;
			j++;
		}
	}

	return S_OK;
}

CAPI_FUNC(HRESULT) TissueForge::engine_del_particle(struct engine *e, int pid)
{
    TF_Log(LOG_DEBUG) << "time: " << e->time * e->dt << ", deleting particle id: " << pid;

    if(pid < 0 || pid >= e->s.size_parts) {
        return error(MDCERR_id);
    }

    Particle *part = e->s.partlist[pid];

    if(part == NULL) {
        return error(MDCERR_null);
    }

    ParticleType *type = &e->types[part->typeId];

    HRESULT hr = type->del_part(pid);
    if(!SUCCEEDED(hr)) {
        return hr;
    }

    std::vector<int32_t> bonds = Bond_IdsForParticle(pid);

    for(int i = 0; i < bonds.size(); ++i) {
        Bond_Destroy(&_Engine.bonds[bonds[i]]);
    }

	std::vector<int32_t> angles = Angle_IdsForParticle(pid);

	for(int i = 0; i < angles.size(); ++i) {
		Angle_Destroy(&_Engine.angles[angles[i]]);
	}

	std::vector<int32_t> dihedrals = Dihedral_IdsForParticle(pid);

	for(int i = 0; i < dihedrals.size(); ++i) {
		Dihedral_Destroy(&_Engine.dihedrals[dihedrals[i]]);
	}

	e->pids_avail.insert(pid);

    return space_del_particle(&e->s, pid);
}

FVector3 TissueForge::engine_origin() {
	return {
        _Engine.s.origin[0],
        _Engine.s.origin[1],
        _Engine.s.origin[2]
    };
}

FVector3 TissueForge::engine_dimensions() {
	return {
        _Engine.s.dim[0],
        _Engine.s.dim[1],
        _Engine.s.dim[2]
    };
}

FVector3 TissueForge::engine_center() {
    return engine_dimensions() / 2.;
}

HRESULT TissueForge::engine_reset(struct engine *e) {
    
    ParticleList parts = ParticleList::all();
    
    HRESULT hr;
    
    for(int i = 0; i < parts.nr_parts; ++i) {
        if(FAILED(hr = engine_del_particle(e, parts.parts[i]))) {
            return hr;
        }
    }
    
    /* all is well... */
    return S_OK;
}
