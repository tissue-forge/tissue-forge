/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2013 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

/* Include configuration header */
#include <mdcore_config.h>

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <limits.h>

/* Include some conditional headers. */
#include <mdcore_config.h>
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef HAVE_SETAFFINITY
    #include <sched.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include <tf_cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfParticle.h>
#include <tfSpace_cell.h>
#include <tfSpace.h>
#include <tfPotential.h>
#include "tf_potential_eval.h"
#include <tfEngine.h>
#include <tfRunner.h>


using namespace TissueForge;


extern unsigned int runner_rcount;


TF_FLATTEN HRESULT TissueForge::runner_dosort(struct runner *r, struct space_cell *c, int flags) {

    struct Particle *p;
    struct space *s;
    int i, k, sid;
    struct Particle *parts;
    struct engine *eng;
    unsigned int *iparts;
    FPTYPE dscale;
    FPTYPE shiftn[3], bias;
    int count;
    
    
    /* break early if one of the cells is empty */
    count = c->count;
    if(count == 0)
        return S_OK;
    
    /* get the space and cutoff */
    eng = r->e;
    s = &(eng->s);
    bias = sqrt(s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2]);
    dscale = (FPTYPE)SHRT_MAX / (2 * bias);
    
    
    /* Make local copies of the parts if requested. */
    if(r->e->flags & engine_flag_localparts) {
        parts = (struct Particle *)alloca(sizeof(struct Particle) * count);
        memcpy(parts, c->parts, sizeof(struct Particle) * count);
    }
    else
        parts = c->parts;
        

    /* Loop over the sort directions. */
    for(sid = 0 ; sid < 13 ; sid++) {
    
        /* In the mask? */
        if(!(flags & (1 << sid)))
            continue;

        /* Get the normalized shift. */
        for(k = 0 ; k < 3 ; k++)
            shiftn[k] = cell_shift[ 3*sid + k ];

        /* Get the pointers to the sorted particle data. */
        iparts = &c->sortlist[ count * sid ];

        /* start by filling the particle ids and dists */
        for(i = 0 ; i < count ; i++) {
            p = &(parts[i]);
            iparts[i] = (i << 16) |
                (unsigned int)(dscale * (bias + p->x[0]*shiftn[0] + p->x[1]*shiftn[1] + p->x[2]*shiftn[2]));
        }

        /* Sort this data in descending order. */
        runner_sort_descending(iparts, count);
    
    }

    /* since nothing bad happened to us... */
    return S_OK;
}
