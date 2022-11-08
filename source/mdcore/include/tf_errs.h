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
 * @file tf_errs.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TF_ERRS_H_
#define _MDCORE_INCLUDE_TF_ERRS_H_
#include "tf_platform.h"


namespace TissueForge {


    enum MDCErrorCode : int {
        MDCERR_ok = 0,
        MDCERR_io,
        MDCERR_null,
        MDCERR_malloc,
        MDCERR_range,
        MDCERR_space,
        MDCERR_pthread,
        MDCERR_runner,
        MDCERR_cell,
        MDCERR_domain,
        MDCERR_nompi,
        MDCERR_mpi,
        MDCERR_reader,
        MDCERR_potential,
        MDCERR_cuda,
        MDCERR_nocuda,
        MDCERR_cudasp,
        MDCERR_maxparts,
        MDCERR_queue,
        MDCERR_rigid,
        MDCERR_subengine,
        MDCERR_id,
        MDCERR_angsspot,
        MDCERR_dihsspot,
        MDCERR_engine,
        MDCERR_spe,
        MDCERR_mfc,
        MDCERR_unavail,
        MDCERR_fifo,
        MDCERR_verlet_overflow,
        MDCERR_tasktype,
        MDCERR_maxpairs,
        MDCERR_nrtasks,
        MDCERR_task,
        MDCERR_taskmaxunlock,
        MDCERR_notcluster,
        MDCERR_wrongptype,
        MDCERR_initorder,
        MDCERR_bond,
        MDCERR_angle,
        MDCERR_dihedral,
        MDCERR_exclusion,
        MDCERR_sets,
        MDCERR_toofast,
        MDCERR_index,
        MDCERR_large_state,
        MDCERR_min_types,
        MDCERR_bad_el_input,
        MDCERR_verify,
        MDCERR_badprop,
        MDCERR_particle,
        MDCERR_nyi,
        MDCERR_ivalsmax,
        MDCERR_fullqueue,
        MDCERR_lock,
        MDCERR_force,
        MDCERR_LAST
    };

    /* list of error messages. */
    static const char *errs_err_msg[MDCERR_LAST] = {
        "All is well.",
        "An IO-error has occurred.",
        "An unexpected NULL pointer was encountered.",
        "A call to malloc failed, probably due to insufficient memory.",
		"One or more values were outside of the allowed range.",
        "An error occured when calling a space function.",
        "A call to a pthread routine failed.",
        "An error occured when calling a runner function.",
        "An error occured while calling a cell function.",
        "The computational domain is too small for the requested operation.",
        "mdcore was not compiled with MPI.",
        "An error occured while calling an MPI function.",
        "An error occured when calling a reader function.",
        "An error occured when calling a potential function.",
        "An error occured when calling a CUDA funtion.",
        "mdcore was not compiled with CUDA support.",
        "CUDA support is only available in single-precision.",
        "Max. number of parts per cell exceeded.",
        "An error occured when calling a queue funtion.",
        "An error occured when evaluating a rigid constraint.",
        "An error occured when calling a subengine.",
        "Invalid id.",
        "Angles do not support scaled or shifted potentials.",
        "Dihedrals do not support scaled or shifted potentials.",
        "An error occured when calling an engine function.",
        "An error occured when calling an SPE function.",
        "An error occured with the memory flow controler.",
        "The requested functionality is not available.",
        "An error occured when calling an fifo function.",
        "Error filling Verlet list: too many neighbours.",
        "Unknown task type.",
        "Too many pairs associated with a single particle in Verlet list.",
        "Task list too short.",
        "An error occured when calling a task function.",
        "Attempted to add an unlock to a full task.",
        "The particle is not a cluster.",
        "An attempt was made to add a particle to a cluster with incompatible type.",
        "Engine types already set, or not initialized in correct order.",
        "An error occured when calling a bond function.",
        "An error occured when calling an angle function.",
        "An error occured when calling a dihedral function.",
        "An error occured when calling an exclusion function.",
        "An error occured while computing the bonded sets.",
        "Particles moving too fast",
        "Invalid index.",
        "An invalid particle state change occurred (large particle).",
        "Must have at least space for 3 particle types.",
        "Inconsistent element inputs.",
        "Failed to verify a particle.",
        "Bad property detected.",
        "An error occured when calling a particle function.",
		"Not yet implemented.",
		"Maximum number of intervals reached before tolerance satisfied.",
        "Attempted to insert into a full queue.",
        "An error occured in a lock function.",
        "An error occured in a force function."
    };

};

#endif // _MDCORE_INCLUDE_TF_ERRS_H_