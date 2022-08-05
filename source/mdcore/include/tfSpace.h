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
 * @file tfSpace.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFSPACE_H_
#define _MDCORE_INCLUDE_TFSPACE_H_
#include <mdcore_config.h>
#include "tfSpace_cell.h"

#include <vector>


/* space error codes */
#define space_err_ok                    0
#define space_err_null                  -1
#define space_err_malloc                -2
#define space_err_cell                  -3
#define space_err_pthread               -4
#define space_err_range                 -5
#define space_err_maxpairs              -6
#define space_err_nrtasks               -7
#define space_err_task                  -8
#define space_err_invalid_partid        -9

#define space_partlist_incr             100

/** Maximum number of cells per tuple. */
#define space_maxtuples                 4

/** Maximum number of interactions per particle in the Verlet list. */
#define space_verlet_maxpairs           800

/* some useful macros */
/** Converts the index triplet (@c i, @c j, @c k) to the cell id in the
    #space @c s. */
#define space_cellid(s,i,j,k)           (  ((i)*(s)->cdim[1] + (j)) * (s)->cdim[2] + (k) )
#define celldims_cellid(cdim,i,j,k)     (  ((i)*cdim[1] + (j)) * cdim[2] + (k) )

/** Convert tuple ids into the pairid index. */
#define space_pairind(i,j)              ( space_maxtuples*(i) - (i)*((i)+1)/2 + (j) )


namespace TissueForge { 


    /* some constants */
    enum PeriodicFlags {
        space_periodic_none       = 0,
        space_periodic_x          = 1 << 0,
        space_periodic_y          = 1 << 1,
        space_periodic_z          = 1 << 2,
        space_periodic_full       = (1 << 0) | (1 << 1) | (1 << 2),
        space_periodic_ghost_x    = 1 << 3,
        space_periodic_ghost_y    = 1 << 4,
        space_periodic_ghost_z    = 1 << 5,
        space_periodic_ghost_full = (1 << 3) | (1 << 4) | (1 << 5),
        SPACE_FREESLIP_X          = 1 << 6,
        SPACE_FREESLIP_Y          = 1 << 7,
        SPACE_FREESLIP_Z          = 1 << 8,
        SPACE_FREESLIP_FULL       = (1 << 6) | (1 << 7) | (1 << 8),
    };


    /** ID of the last error */
    CAPI_DATA(int) space_err;

    /** Struct for Verlet list entries. */
    struct verlet_entry {

        /** The particle. */
        struct Particle *p;

        /** The interaction potential. */
        struct Potential *pot;

        /** The integer shift relative to this particle. */
        signed char shift[3];

    };

    /** Struct for each cellpair (see #space_getpair). */
    struct cellpair {

        /** Indices of the cells involved. */
        int i, j;

        /** Relative shift between cell centres. */
        FPTYPE shift[3];

        /** Pairwise Verlet stuff. */
        int size, count;
        short int *pairs;
        short int *nr_pairs;

        /** Pointer to chain pairs together. */
        struct cellpair *next;

    };

    /** Struct for groups of cellpairs. */
    struct celltuple {

        /** IDs of the cells in this tuple. */
        int cellid[ space_maxtuples ];

        /** Nr. of cells in this tuple. */
        int n;

        /** Ids of the underlying pairs. */
        int pairid[ space_maxtuples * (space_maxtuples + 1) / 2 ];

        /* Buffer value to keep the size at 64 bytes for space_maxtuples==4. */
        int buff;

    };

    /**
     * The space structure
     */
    typedef struct space {
        /** Real dimensions. */
        FPTYPE dim[3];

        /** Location of origin. */
        FPTYPE origin[3];

        /** Space dimension in cells. */
        int cdim[3];

        /** Number of cells within cutoff in each dimension. */
        int span[3];

        /** Cell edge lengths and their inverse. */
        FPTYPE h[3], ih[3];

        /** The cutoff and the cutoff squared. */
        FPTYPE cutoff, cutoff2;

        /** Periodicities. */
        unsigned int period;

        /** Total nr of cells in this space. */
        int nr_cells;

        /** IDs of real, ghost and marked cells. */
        int *cid_real, *cid_ghost, *cid_marked;
        int nr_real, nr_ghost, nr_marked;

        /** Array of cells spanning the space. */
        struct space_cell *cells;

        /** The total number of tasks. */
        int nr_tasks, tasks_size;

        /** Array of tasks. */
        struct task *tasks;

        /** Condition/mutex to signal task availability. */
        pthread_mutex_t tasks_mutex;
        pthread_cond_t tasks_avail;

        /** Taboo-list for collision avoidance */
        char *cells_taboo;

        /** Id of #runner owning each cell. */
        char *cells_owner;

        /** Counter for the number of swaps in every step. */
        int nr_swaps, nr_stalls;

        /** Array of pointers to the individual parts, sorted by their ID. */
        struct Particle **partlist;

        /** store the large particles in the large parts cell, its special */
        space_cell largeparts;

        /** Array of pointers to the #cell of individual parts, sorted by their ID. */
        struct space_cell **celllist;

        /** Number of parts in this space and size of the buffers partlist and celllist. */
        int nr_parts, size_parts;

        /**
         * number of visible particles and large particles.
         * Yes... mixing rendering and simulation, but put it here
         * so we only have to go through the list once to get this count.
         *
         * updated by engine_advance
         */
        int nr_visible_parts;
        int nr_visible_large_parts;

        /** Trigger re-building the cells/sorts. */
        int verlet_rebuild;

        /** The maximum particle displacement over all cells. */
        FPTYPE maxdx;

        /** Potential energy collected by the space itself. */
        FPTYPE epot, epot_nonbond, epot_bond, epot_angle, epot_dihedral, epot_exclusion;

        /** Data for the verlet list. */
        struct verlet_entry *verlet_list;
        FPTYPE *verlet_oldx, verlet_maxdx;
        int *verlet_nrpairs;
        int verlet_size, verlet_next;
        pthread_mutex_t verlet_force_mutex;

        /** The total number of cell pairs. */
        int nr_pairs;

        /** Array of cell pairs. */
        struct cellpair *pairs;

        /** Id of the next unprocessed pair (for #space_getpair) */
        int next_pair;

        /** Array of cell tuples. */
        struct celltuple *tuples;

        /** The number of tuples. */
        int nr_tuples;

        /** The ID of the next free tuple or cell. */
        int next_tuple, next_cell;

        /** Mutex for accessing the cell pairs. */
        pthread_mutex_t cellpairs_mutex;

        /** Spin-lock for accessing the cell pairs. */
        //lock_type cellpairs_spinlock;

        /** Condition to wait for free cells on. */
        pthread_cond_t cellpairs_avail;

    } space;


    /* associated functions */
    CAPI_FUNC(int) space_init(
        struct space *s, 
        const FPTYPE *origin,
        const FPTYPE *dim, 
        FPTYPE *L, 
        FPTYPE cutoff, 
        const struct BoundaryConditions *bc
    );

    CAPI_FUNC(int) space_getsid(
        struct space *s, 
        struct space_cell **ci,
        struct space_cell **cj, 
        FPTYPE *shift
    );

    CAPI_FUNC(int) space_shuffle(struct space *s);
    CAPI_FUNC(int) space_shuffle_local(struct space *s);

    CAPI_FUNC(int) space_growparts(struct space *s, unsigned int size_incr);

    /**
     * @brief Add a #part to a #space at the given coordinates. The given
     * particle p is only used for the attributes, it itself is not added,
     * rather a new memory block is allocated, and the contents of p
     * get copied in there.
     *
     * @param s The space to which @c p should be added.
     * @param p The #part to be added.
     * @param x A pointer to an array of three FPTYPEs containing the particle
     *      position.
     * @param result pointer to the newly allocated particle.
     *
     * @returns #space_err_ok or < 0 on error (see #space_err).
     *
     * Inserts a #part @c p into the #space @c s at the position @c x.
     * Note that since particle positions in #part are relative to the cell, that
     * data in @c p is overwritten and @c x is used.
     *
     * This is a PRIVATE function, literally only the engine should call this.
     * Does NOT manage ref count on particle types in the engine.
     */
    CAPI_FUNC(int) space_addpart(
        struct space *s, 
        struct Particle *p,
        FPTYPE *x, 
        struct Particle **result
    );

    CAPI_FUNC(int) space_addparts(
        struct space *s, 
        int nr_parts, 
        struct Particle **parts, 
        FPTYPE **xparts
    );

    /**
     * get the cell id for a position,
     * negative on failure
     *
     * @param s the #space
     * @param x the position
     * @param cellids [optional] get the (i,j,k) indices of the space cell.
     * returns the absolute cell id (index into array).
     */
    CAPI_FUNC(int) space_get_cellids_for_pos(struct space *s, FPTYPE *x, int *cellids);

    /**
     * Deletes a particle from the space, and sets the partlist[pid] to null.
     *
     * this will decrement the python pointer in p, and overwrite the memeory
     * pointed to by p. Any pointer to this will no longer be valid.
     *
     * Note, pid is the global particle id, and is the index in partlist of the
     * particle.
     */
    CAPI_FUNC(HRESULT) space_del_particle(struct space *s,  int pid);


    /**
     * A style was changed, so need to update any counts the
     * space object has.
     */
    CAPI_FUNC(HRESULT) space_update_style(struct space *s);

    CAPI_FUNC(int) space_prepare(struct space *s);
    CAPI_FUNC(int) space_getpos(struct space *s, int id, FPTYPE *x);
    CAPI_FUNC(int) space_setpos(struct space *s, int id, FPTYPE *x);
    CAPI_FUNC(int) space_flush(struct space *s);
    CAPI_FUNC(int) space_flush_ghosts(struct space *s);
    CAPI_FUNC(struct task*) space_addtask(
        struct space *s, 
        int type,
        int subtype, 
        int flags, 
        int i, 
        int j
    );


    CAPI_FUNC(int) space_verlet_init(struct space *s, int list_global);
    CAPI_FUNC(int) space_gettuple(struct space *s, struct celltuple **out, int wait);
    CAPI_FUNC(int) space_getcell(struct space *s, struct space_cell **out);
    CAPI_FUNC(int) space_verlet_force(struct space *s, FPTYPE *f, FPTYPE epot);
    CAPI_FUNC(int) space_releasepair(struct space *s, int ci, int cj);

};

#endif // _MDCORE_INCLUDE_TFSPACE_H_