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
 * @file tfSpace.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFSPACE_H_
#define _MDCORE_INCLUDE_TFSPACE_H_
#include <mdcore_config.h>
#include "tfSpace_cell.h"

#include <vector>


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

    /**
     * @brief Initialize the space with the given dimensions.
     *
     * @param s The #space to initialize.
     * @param origin Pointer to an array of three FPTYPEs specifying the origin
     *      of the rectangular domain.
     * @param dim Pointer to an array of three FPTYPEs specifying the length
     *      of the rectangular domain along each dimension.
     * @param L The minimum cell edge length, in each dimension.
     * @param cutoff A FPTYPE-precision value containing the maximum cutoff lenght
     *      that will be used in the potentials.
     * @param period Unsigned integer containing the flags #space_periodic_x,
     *      #space_periodic_y and/or #space_periodic_z or #space_periodic_full.
     *
     * This routine initializes the fields of the #space @c s, creates the cells and
     * generates the cell-pair list.
     */
    CAPI_FUNC(HRESULT) space_init(
        struct space *s, 
        const FPTYPE *origin,
        const FPTYPE *dim, 
        FPTYPE *L, 
        FPTYPE cutoff, 
        const struct BoundaryConditions *bc
    );

    /** 
     * @brief Get the sort-ID and flip the cells if necessary.
     *
     * @param s The #space.
     * @param ci FPTYPE pointer to the first #cell.
     * @param cj FPTYPE pointer to the second #cell.
     */
    CAPI_FUNC(int) space_getsid(
        struct space *s, 
        struct space_cell **ci,
        struct space_cell **cj, 
        FPTYPE *shift
    );

    /**
     * @brief Run through the cells of a #space and make sure every particle is in
     * its place.
     *
     * @param s The #space on which to operate.
     *
     * Runs through the cells of @c s and if a particle has stepped outside the
     * cell bounds, moves it to the correct cell.
     */
    CAPI_FUNC(HRESULT) space_shuffle(struct space *s);

    /**
     * @brief Run through the non-ghost cells of a #space and make sure every
     * particle is in its place.
     *
     * @param s The #space on which to operate.
     *
     * Runs through the cells of @c s and if a particle has stepped outside the
     * cell bounds, moves it to the correct cell.
     */
    CAPI_FUNC(HRESULT) space_shuffle_local(struct space *s);

    /**
     * @brief Grow the parts allocated to a #space
     * 
     * @param s The #space on which to operate.
     * @param size_incr The increment in size.
     */
    CAPI_FUNC(HRESULT) space_growparts(struct space *s, unsigned int size_incr);

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
     * Inserts a #part @c p into the #space @c s at the position @c x.
     * Note that since particle positions in #part are relative to the cell, that
     * data in @c p is overwritten and @c x is used.
     *
     * This is a PRIVATE function, literally only the engine should call this.
     * Does NOT manage ref count on particle types in the engine.
     */
    CAPI_FUNC(HRESULT) space_addpart(
        struct space *s, 
        struct Particle *p,
        FPTYPE *x, 
        struct Particle **result
    );

    CAPI_FUNC(HRESULT) space_addparts(
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

    /**
     * @brief Prepare the tasks before a time step.
     *
     * @param s A pointer to the #space to prepare.
     *
     * Initializes the tasks of a #space for a single time step. 
     */
    CAPI_FUNC(HRESULT) space_prepare_tasks(struct space *s);

    /**
     * @brief Prepare the space before a time step.
     *
     * @param s A pointer to the #space to prepare.
     *
     * Initializes a #space for a single time step. This routine runs
     * through the particles and sets their forces to zero.
     */
    CAPI_FUNC(HRESULT) space_prepare(struct space *s);

    /**
     * @brief Get the absolute position of a particle
     *
     * @param s The #space in which the particle resides.
     * @param id The local id of the #part.
     * @param x A pointer to a vector of at least three @c FPTYPEs in
     *      which to store the particle position.
     *
     */
    CAPI_FUNC(HRESULT) space_getpos(struct space *s, int id, FPTYPE *x);

    /**
     * @brief Set the absolute position of a particle. 
     * 
     * @param s The #space in which the particle resides.
     * @param id The local id of the #part.
     * @param x A pointer to a vector of at least three @c FPTYPEs in
     *      which to store the particle position.
     */
    CAPI_FUNC(HRESULT) space_setpos(struct space *s, int id, FPTYPE *x);

    /**
     * @brief Clear all particles from this #space.
     *
     * @param s The #space to flush.
     */
    CAPI_FUNC(HRESULT) space_flush(struct space *s);

    /**
     * @brief Clear all particles from the ghost cells in this #space.
     *
     * @param s The #space to flush.
     */
    CAPI_FUNC(HRESULT) space_flush_ghosts(struct space *s);

    /**
     * @brief Add a task to the given space.
     *
     * @param s The #space.
     * @param type The task type.
     * @param subtype The task subtype.
     * @param flags The task flags.
     * @param i Index of the first cell/domain.
     * @param j Index of the second cell/domain.
     *
     * @return A pointer to the newly added #task or @c NULL if anything went wrong.
     */
    CAPI_FUNC(struct task*) space_addtask(
        struct space *s, 
        int type,
        int subtype, 
        int flags, 
        int i, 
        int j
    );

    /**
     * @brief Initialize the Verlet-list data structures.
     *
     * @param s The #space.
     */
    CAPI_FUNC(HRESULT) space_verlet_init(struct space *s, int list_global);

    /**
     * @brief Get the next free #celltuple from the space.
     *
     * @param s The #space in which to look for tuples.
     * @param out A pointer to a #celltuple in which to copy the result.
     * @param wait A boolean value specifying if to wait for free tuples
     *      or not.
     *
     * @return The number of #celltuple found or 0 if the list is empty and
     *      < 0 on error.
     */
    CAPI_FUNC(int) space_gettuple(struct space *s, struct celltuple **out, int wait);

    /**
     * @brief Get the next unprocessed cell from the spaece.
     *
     * @param s The #space.
     * @param out Pointer to a pointer to #cell in which to store the results.
     *
     * @return @c 1 if a cell was found, 0 if the list is empty
     *      or < 0 on error.
     */
    CAPI_FUNC(int) space_getcell(struct space *s, struct space_cell **out);

    /**
     * @brief Collect forces and potential energies
     *
     * @param s The #space.
     * @param maxcount The maximum number of entries.
     * @param from Pointer to an integer which will contain the index to the
     *        first entry on success.
     * @param to Pointer to an integer which will contain the index to the
     *        last entry on success.
     *
     * @return The number of entries returned or < 0 on error.
     */
    CAPI_FUNC(int) space_verlet_force(struct space *s, FPTYPE *f, FPTYPE epot);

    /**
     * @brief Free the cells involved in the current pair.
     *
     * @param s The #space to operate on.
     * @param ci ID of the first cell.
     * @param cj ID of the second cell.
     *
     * Decreases the taboo-counter of the cells involved in the pair
     * and signals any #runner that might be waiting.
     * Note that only a single waiting #runner is released per released cell
     * and therefore, if two different cells become free, the condition
     * @c cellpairs_avail is signaled twice.
     */
    CAPI_FUNC(HRESULT) space_releasepair(struct space *s, int ci, int cj);

};

#endif // _MDCORE_INCLUDE_TFSPACE_H_