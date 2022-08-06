/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2013 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include <cycle.h>
#include <tf_errs.h>
#include <tf_fptype.h>
#include <tf_lock.h>
#include <tfTask.h>


using namespace TissueForge;


/* Global variables. */
/** The ID of the last error. */
int TissueForge::task_err = task_err_ok;

/* the error macro. */
#define error(id)				(task_err = errs_register(id, task_err_msg[-(id)], __LINE__, __FUNCTION__, __FILE__))

/* list of error messages. */
const char *task_err_msg[4] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "Attempted to add an unlock to a full task.",
};

/**
 * @brief Add a task dependency.
 * 
 * @param ta The unlocking #task.
 * @param tb The unlocked #task.
 *
 */
 
int TissueForge::task_addunlock(struct task *ta, struct task *tb) {

    /* Is there space for this? */
    if(ta->nr_unlock >= task_max_unlock)
        return error(task_err_maxunlock);

    /* Add the unlock. */
    ta->unlock[ ta->nr_unlock ] = tb;
    ta->nr_unlock += 1;
    
    /* Ta-da! */
    return task_err_ok;
    
}

std::ostream& operator<<(std::ostream& os, const struct task* t) {

    /** Task type/subtype. */
    os << "task { type: ";
    
    switch(t->type) {
        case task_type_none:
            os << "none";
            break;
        case task_type_self:
            os << "self";
            break;
        case task_type_pair:
            os << "pair";
            break;
        case  task_type_sort:
            os << "sort";
            break;
        case   task_type_bonded:
            os << "bonded";
            break;
        case  task_type_count:
            os << "count";
            break;
        default:
            os << "error";
            break;
    }
    
    os << ", subtype: " << t->subtype;
    
    os << ", i: " << t->i << ", j: " << t->j ;
    
    os << ", unlocks: " << t->nr_unlock << "}" << std::endl;
    
    return os;
}

    
