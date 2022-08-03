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

#ifndef _MDCORE_INCLUDE_TFENGINE_H_
#define _MDCORE_INCLUDE_TFENGINE_H_

#include "tf_platform.h"
#include <pthread.h>
#include "tfSpace.h"
#include "cycle.h"
#include "tfBoundaryConditions.h"
#include <tfSubEngine.h>
#include <mutex>
#include <thread>
#include <set>


/* engine error codes */
#define engine_err_ok                    0
#define engine_err_null                  -1
#define engine_err_malloc                -2
#define engine_err_space                 -3
#define engine_err_pthread               -4
#define engine_err_runner                -5
#define engine_err_range                 -6
#define engine_err_cell                  -7
#define engine_err_domain                -8
#define engine_err_nompi                 -9
#define engine_err_mpi                   -10
#define engine_err_bond                  -11
#define engine_err_angle                 -12
#define engine_err_reader                -13
#define engine_err_psf                   -14
#define engine_err_pdb                   -15
#define engine_err_cpf                   -16
#define engine_err_potential             -17
#define engine_err_exclusion             -18
#define engine_err_sets                  -19
#define engine_err_dihedral              -20
#define engine_err_cuda                  -21
#define engine_err_nocuda                -22
#define engine_err_cudasp                -23
#define engine_err_maxparts              -24
#define engine_err_queue                 -25
#define engine_err_rigid                 -26
#define engine_err_cutoff		 		 -27
#define engine_err_nometis				 -28
#define engine_err_toofast               -29
#define engine_err_subengine 			 -30

#define engine_bonds_chunk               100
#define engine_angles_chunk              100
#define engine_rigids_chunk              50
#define engine_dihedrals_chunk           100
#define engine_exclusions_chunk          100
#define engine_readbuff                  16384
#define engine_maxgpu                    10
#define engine_pshake_steps              20
#define engine_maxKcutoff                2

#define engine_split_MPI		1
#define engine_split_GPU		2

#define engine_bonded_maxnrthreads       16
#define engine_bonded_nrthreads          ((omp_get_num_threads()<engine_bonded_maxnrthreads)?omp_get_num_threads():engine_bonded_maxnrthreads)

#ifdef MDCORE_MAXNRTYPES
#define engine_maxnrtypes 				 MDCORE_MAXNRTYPES
#else
#define engine_maxnrtypes				 128
#endif


namespace TissueForge { 


	/* some constants */
	enum EngineFlags {
		engine_flag_none                 = 0,
		engine_flag_static               = 1 << 0,
		engine_flag_localparts           = 1 << 1,
		engine_flag_cuda                 = 1 << 2,
		engine_flag_explepot             = 1 << 3,
		engine_flag_verlet               = 1 << 4,
		engine_flag_verlet_pairwise      = 1 << 5,
		engine_flag_affinity             = 1 << 6,
		engine_flag_prefetch             = 1 << 7,
		engine_flag_verlet_pseudo        = 1 << 8,
		engine_flag_unsorted             = 1 << 9,
		engine_flag_shake                = 1 << 10,
		engine_flag_mpi                  = 1 << 11,
		engine_flag_parbonded            = 1 << 12,
		engine_flag_async                = 1 << 13,
		engine_flag_sets                 = 1 << 14,
		engine_flag_nullpart             = 1 << 15,
		engine_flag_initialized          = 1 << 16,
		engine_flag_velocity_clamp       = 1 << 17,
	};

	enum EngineIntegrator {
		FORWARD_EULER,
		RUNGE_KUTTA_4
	};

	/** Timmer IDs. */
	enum {
		engine_timer_step = 0,
		engine_timer_kinetic, 
		engine_timer_prepare,
		engine_timer_verlet,
		engine_timer_exchange1,
		engine_timer_nonbond,
		engine_timer_bonded,
		engine_timer_bonded_sort,
		engine_timer_bonds,
		engine_timer_angles,
		engine_timer_dihedrals,
		engine_timer_exclusions,
		engine_timer_advance,
		engine_timer_rigid,
		engine_timer_exchange2,
		engine_timer_shuffle,
		engine_timer_cuda_load,
		engine_timer_cuda_unload,
		engine_timer_cuda_dopairs,
		engine_timer_render,
		engine_timer_image_data,
		engine_timer_render_total,
		engine_timer_total,
		engine_timer_last
	};


	/** Timmer IDs. */
	enum {
		ENGINE_TIMER_STEP           = 1 << 0,
		ENGINE_TIMER_PREPARE        = 1 << 1,
		ENGINE_TIMER_VERLET         = 1 << 2,
		ENGINE_TIMER_EXCHANGE1      = 1 << 3,
		ENGINE_TIMER_NONBOND        = 1 << 4,
		ENGINE_TIMER_BONDED         = 1 << 5,
		ENGINE_TIMER_BONDED_SORT    = 1 << 6,
		ENGINE_TIMER_BONDS          = 1 << 7,
		ENGINE_TIMER_ANGLES         = 1 << 8,
		ENGINE_TIMER_DIHEDRALS      = 1 << 9,
		ENGINE_TIMER_EXCLUSIONS     = 1 << 10,
		ENGINE_TIMER_ADVANCE        = 1 << 11,
		ENGINE_TIMER_RIGID          = 1 << 12,
		ENGINE_TIMER_EXCHANGE2      = 1 << 13,
		ENGINE_TIMER_SHUFFLE        = 1 << 14,
		ENGINE_TIMER_CUDA_LOAD      = 1 << 15,
		ENGINE_TIMER_CUDA_UNLOAD    = 1 << 16,
		ENGINE_TIMER_CUDA_DOPAIRS   = 1 << 17,
		ENGINE_TIMER_RENDER         = 1 << 18,
		ENGINE_TIMER_LAST           = 1 << 19
	};

	enum {
		// forces that set the persistent_force should
		// update values now. Otherwise, the integrator is
		// probably in a multi-step and should use the saved
		// value
		INTEGRATOR_UPDATE_PERSISTENTFORCE    = 1 << 0
	};


	/** ID of the last error. */
	CAPI_DATA(int) engine_err;

	/** List of error messages. */
	CAPI_DATA(const char *) engine_err_msg[];

	struct CustomForce;

	/**
	 * The #engine structure.
	 */
	typedef struct CAPI_EXPORT engine {

		/** Some flags controlling how this engine works. */
		unsigned int flags;

		/**
		 * Internal flags related to multi-step integrators,
		 */
		unsigned int integrator_flags;

	#ifdef HAVE_CUDA
		/** Some flags controlling which cuda scheduling we use. */
		unsigned int flags_cuda;
	#endif

		/** The space on which to work */
		struct space s;

		/** Time variables */
		long time;
		FPTYPE dt;

		FPTYPE temperature;

		// Boltzmann constant
		FPTYPE K;

		/** TODO, clean up this design for types and static engine. */
		/** What is the maximum nr of types? */
		static const int max_type;
		static int nr_types;

		/** The particle types. */
		static struct ParticleType *types;

		/** The interaction matrix */
		struct Potential **p, **p_cluster;

		/**
		 * vector of forces for types, indexed
		 * by type id.
		 */
		struct Force **forces;

		/**
		 * interaction matrix of pointers to fluxes, same layout as
		 * potential matrix p.
		 */
		struct Fluxes **fluxes;

		/** Mutexes, conditions and counters for the barrier */
		pthread_mutex_t barrier_mutex;
		pthread_cond_t barrier_cond;
		pthread_cond_t done_cond;
		int barrier_count;

		/** Nr of runners */
		int nr_runners;

		/** The runners */
		struct runner *runners;

		/** The queues for the runners. */
		struct queue *queues;
		int nr_queues;

		/** The ID of the computational node we are on. */
		int nodeID;
		int nr_nodes;

		/** Lists of cells to exchange with other nodes. */
		struct engine_comm *send, *recv;

		/** Recycled particle ids */
		std::set<unsigned int> pids_avail;

		/** List of bonds. */
		struct Bond *bonds;

		/**
		 * total number of bonds, active or not.
		 */
		int nr_bonds;

		/**
		 * number of active bonds.
		 * note, active bonds are not necessarily in contigous order.
		 */
		int nr_active_bonds;

		/**
		 * allocate size of bonds array
		 */
		int bonds_size;

		// mutex for anything that modifies the *number* of bonds.
		std::mutex bonds_mutex;

		/** List of exclusions. */
		struct exclusion *exclusions;

		/** Nr. of exclusions. */
		int nr_exclusions, exclusions_size;

		/** List of rigid bodies. */
		struct rigid *rigids;

		/** List linking parts to rigids. */
		int *part2rigid;

		/** Nr. of rigids. */
		int nr_rigids, rigids_size, nr_constr, rigids_local, rigids_semilocal;

		/** Rigid solver tolerance. */
		FPTYPE tol_rigid;

		/** List of angles. */
		struct Angle *angles;

		/**
		 * total number of angles, active or not.
		 */
		int nr_angles;

		/** 
		 * number of active angles. 
		 * note, active angles are not necessarily in contiguous order
		 */
		int nr_active_angles;

		/** Allocated size of angles array */
		int angles_size;

		/** List of dihedrals. */
		struct Dihedral *dihedrals;

		/**
		 * total number of dihedrals, active or not.
		 */
		int nr_dihedrals;

		/** 
		 * number of active dihedrals. 
		 * note, active dihedrals are not necessarily in contiguous order
		 */
		int nr_active_dihedrals;

		/** Allocated size of dihedrals array */
		int dihedrals_size;

		/** The Comm object for mpi. */
	#ifdef WITH_MPI
		MPI_Comm comm;
		pthread_mutex_t xchg_mutex;
		pthread_cond_t xchg_cond;
		short int xchg_started, xchg_running;
		pthread_t thread_exchg;
		pthread_mutex_t xchg2_mutex;
		pthread_cond_t xchg2_cond;
		short int xchg2_started, xchg2_running;
		pthread_t thread_exchg2;
	#endif

		/** Pointers to device data for CUDA. */
	#ifdef HAVE_CUDA
		void *sortlists_cuda[ engine_maxgpu ];
		int nr_pots_cuda, nr_pots_cluster_cuda, *pind_cuda[ engine_maxgpu ], *pind_cluster_cuda[ engine_maxgpu ], *offsets_cuda[ engine_maxgpu ];
		int nr_devices, devices[ engine_maxgpu ], nr_queues_cuda;
		float *forces_cuda[ engine_maxgpu ];
		void *parts_cuda[ engine_maxgpu ], *part_states_cuda[engine_maxgpu];
		void *parts_cuda_local, *part_states_cuda_local;
		int *cells_cuda_local[ engine_maxgpu];
		int cells_cuda_nr[ engine_maxgpu ];
		int *counts_cuda[ engine_maxgpu ], *counts_cuda_local[ engine_maxgpu ];
		int *ind_cuda[ engine_maxgpu ], *ind_cuda_local[ engine_maxgpu ];
		struct task_cuda *tasks_cuda[ engine_maxgpu ];
		int *taboo_cuda[ engine_maxgpu ];
		int nrtasks_cuda[ engine_maxgpu ];
		void *streams[ engine_maxgpu ];
		int nr_blocks[engine_maxgpu], nr_threads[engine_maxgpu];
		int nr_fluxes_cuda, *fxind_cuda[engine_maxgpu];
		void **fluxes_cuda[engine_maxgpu];
		float *fluxes_next_cuda[engine_maxgpu];
		bool bonds_cuda = false;
		bool angles_cuda = false;
	#endif

		/** Timers. */
		ticks timers[engine_timer_last];

		/** Bonded sets. */
		struct engine_set *sets;
		int nr_sets;
		
		FPTYPE wall_time;
		
		// bitmask of timers to show in performance counter output.
		uint32_t timers_mask;
		
		long timer_output_period;

		/**
		 * vector of constant forces. Because these forces get
		 * updates from user defined functions, we keep a copy of them
		 * here in addtion to the other copy in p_singlebody.
		 */
		std::vector<CustomForce*> custom_forces;

		/**
		 * particle maximum velocity as a fraction of space cell size.
		 * good values for this are around 0.2, meaning that a particle can
		 * move about 1/5th of a cell length per time step.
		 *
		 * if this is set of infinity, means there is not max speed.
		 * if the particle speed exceeds maximum velocity, the velocity
		 * is clamped to this speed.
		 *
		 * defaults to 0.1.
		 */
		FPTYPE particle_max_dist_fraction;

		/**
		 *
		 */
		FPTYPE computed_volume;

		EngineIntegrator integrator;

		BoundaryConditions boundary_conditions;

		/**
		 * @brief Borrowed references to registered subengines.
		 * 
		 */
		std::vector<SubEngine*> subengines;
		
		
		/**
		 * saved objects from init
		 */
		BoundaryConditionsArgsContainer *_init_boundary_conditions;
		int _init_cells[3];
	} engine;


	/**
	 * Structure storing grouped sets of bonded interactions.
	 */
	typedef struct engine_set {

		/* Counts of the different interaction types. */
		int nr_bonds, nr_angles, nr_dihedrals, nr_exclusions, weight;

		/* Lists of ID of the relevant bonded types. */
		struct Bond *bonds;
		struct Angle *angles;
		struct Dihedral *dihedrals;
		struct exclusion *exclusions;

		/* Nr of sets with which this set conflicts. */
		int nr_confl;

		/* IDs of the sets with which this set conflicts. */
		int *confl;

	} engine_set;


	/**
	 * Structure storing which cells to send/receive to/from another node.
	 */
	typedef struct engine_comm {

		/* Size and count of the cellids. */
		int count, size;

		int *cellid;

	} engine_comm;


	/* associated functions */
	CAPI_FUNC(int) engine_addpot(struct engine *e, struct Potential *p, int i, int j);

	CAPI_FUNC(int) engine_addfluxes(struct engine *e, struct Fluxes *f, int i, int j);
	Fluxes *engine_getfluxes(struct engine *e, int i, int j);

	/**
	 * Add a single body force to the engine.
	 */
	CAPI_FUNC(int) engine_add_singlebody_force(struct engine *e, struct Force *p, int typeId);

	/**
	 * allocates a new angle, returns its id.
	 */
	CAPI_FUNC(int) engine_angle_alloc(struct engine *e, Angle **out);

	CAPI_FUNC(int) engine_angle_eval(struct engine *e);
	CAPI_FUNC(int) engine_barrier(struct engine *e);
	CAPI_FUNC(int) engine_bond_eval(struct engine *e);
	CAPI_FUNC(int) engine_bonded_eval(struct engine *e);
	CAPI_FUNC(int) engine_bonded_eval_sets(struct engine *e);
	CAPI_FUNC(int) engine_bonded_sets(struct engine *e, int max_sets);
	CAPI_FUNC(int) engine_dihedral_alloc(struct engine *e, Dihedral **out);
	CAPI_FUNC(int) engine_dihedral_eval(struct engine *e);
	CAPI_FUNC(int) engine_exclusion_add(struct engine *e, int i, int j);
	CAPI_FUNC(int) engine_exclusion_eval(struct engine *e);
	CAPI_FUNC(int) engine_exclusion_shrink(struct engine *e);
	CAPI_FUNC(int) engine_finalize(struct engine *e);
	CAPI_FUNC(int) engine_flush_ghosts(struct engine *e);
	CAPI_FUNC(int) engine_flush(struct engine *e);
	CAPI_FUNC(int) engine_gettype(struct engine *e, char *name);
	CAPI_FUNC(int) engine_gettype2(struct engine *e, char *name2);

	/**
	 * allocates a new bond, returns a pointer to it.
	 * returns index of new object.
	 */
	int engine_bond_alloc (struct engine *e, struct Bond **result);

	/**
	 * External C apps should call this to get a particle type ptr.
	 */
	CAPI_FUNC(struct ParticleType*) engine_type(int id);

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
	 * This is the single, central function that actually allocates particle space,
	 * and inserts a new particle into the engine.
	 *
	 * Increases the ref count on the particle type.
	 */
	CAPI_FUNC(int) engine_addpart(
		struct engine *e, 
		struct Particle *p,
		FPTYPE *x, 
		struct Particle **result
	);

	CAPI_FUNC(int) engine_addparts(struct engine *e, int nr_parts, struct Particle **parts, FPTYPE **x);

	/**
	 * @brief Add a type definition.
	 *
	 * @param e The #engine.
	 * @param mass The particle type mass.
	 * @param charge The particle type charge.
	 * @param name Particle name, can be @c NULL.
	 * @param name2 Particle second name, can be @c NULL.
	 *
	 * @return The type ID or < 0 on error (see #engine_err).
	 *
	 * The particle type ID must be an integer greater or equal to 0
	 * and less than the value @c max_type specified in #engine_init.
	 */
	CAPI_FUNC(int) engine_addtype(
		struct engine *e, 
		FPTYPE mass, 
		FPTYPE charge,
		const char *name, 
		const char *name2
	);


	/**
	 * @brief Initialize an #engine with the given data.
	 *
	 * The number of spatial cells in each cartesion dimension is floor( dim[i] / L[i] ), or
	 * the physical size of the space in that dimension divided by the minimum size size of
	 * each cell.
	 *
	 * @param e The #engine to initialize.
	 * @param origin An array of three FPTYPEs containing the cartesian origin
	 *      of the space.
	 * @param dim An array of three FPTYPEs containing the size of the space.
	 *
	 * @param cells length 3 integer vector of number of cells in each direction.
	 *
	 * @param cutoff The maximum interaction cutoff to use.
	 * @param period A bitmask describing the periodicity of the domain
	 *      (see #space_periodic_full).
	 * @param max_type The maximum number of particle types that will be used
	 *      by this engine.
	 * @param flags Bit-mask containing the flags for this engine.
	 *
	 * @return #engine_err_ok or < 0 on error (see #engine_err).
	 */
	CAPI_FUNC(int) engine_init(
		struct engine *e, 
		const FPTYPE *origin, 
		const FPTYPE *dim, 
		int *cells,
		FPTYPE cutoff, 
		BoundaryConditionsArgsContainer *boundaryConditions, 
		int max_type, 
		unsigned int flags
	);

	/**
	 * clears all the uaer allocated objects, resets to state when created.
	 */
	CAPI_FUNC(int) engine_reset(struct engine *e);

	CAPI_FUNC(int) engine_load_ghosts(
		struct engine *e, 
		FPTYPE *x, 
		FPTYPE *v, 
		int *type, 
		int *pid,
		int *vid, FPTYPE *q, 
		unsigned int *flags, 
		int N
	);
	CAPI_FUNC(int) engine_load(
		struct engine *e, 
		FPTYPE *x, 
		FPTYPE *v, 
		int *type, 
		int *pid, 
		int *vid,
		FPTYPE *charge, unsigned int *flags, int N
	);
	CAPI_FUNC(int) engine_nonbond_eval(struct engine *e);
	CAPI_FUNC(int) engine_rigid_add(struct engine *e, int pid, int pjd, FPTYPE d);
	CAPI_FUNC(int) engine_rigid_eval(struct engine *e);
	CAPI_FUNC(int) engine_rigid_sort(struct engine *e);
	CAPI_FUNC(int) engine_rigid_unsort(struct engine *e);
	CAPI_FUNC(int) engine_shuffle(struct engine *e);
	CAPI_FUNC(int) engine_split_bisect(struct engine *e, int N, int particle_flags);
	CAPI_FUNC(int) engine_split(struct engine *e);

	CAPI_FUNC(int) engine_start(struct engine *e, int nr_runners, int nr_queues);
	CAPI_FUNC(int) engine_step(struct engine *e);
	CAPI_FUNC(int) engine_timers_reset(struct engine *e);
	CAPI_FUNC(int) engine_unload_marked(
		struct engine *e, 
		FPTYPE *x, 
		FPTYPE *v, 
		int *type, 
		int *pid,
		int *vid, 
		FPTYPE *q, 
		unsigned int *flags, 
		FPTYPE *epot, 
		int N
	);
	CAPI_FUNC(int) engine_unload_strays(
		struct engine *e, 
		FPTYPE *x, 
		FPTYPE *v, 
		int *type, 
		int *pid,
		int *vid, 
		FPTYPE *q, 
		unsigned int *flags, 
		FPTYPE *epot, 
		int N
	);
	CAPI_FUNC(int) engine_unload(
		struct engine *e, 
		FPTYPE *x, 
		FPTYPE *v, 
		int *type, 
		int *pid, 
		int *vid,
		FPTYPE *charge, 
		unsigned int *flags, 
		FPTYPE *epot, 
		int N
	);
	CAPI_FUNC(int) engine_verlet_update(struct engine *e);

	/**
	 * gets the next available particle id to use when creating a new particle.
	 */
	CAPI_FUNC(int) engine_next_partid(struct engine *e);

	/**
	 * gets the next available particle ids to use when creating a new particle.
	 */
	CAPI_FUNC(int) engine_next_partids(struct engine *e, int nr_ids, int *ids);

	/**
	 * internal method to clear data before calculating forces on all objects
	 */
	int engine_force_prep(struct engine *e);

	/**
	 * internal method to calculate forces on all objects
	 */
	int engine_force(struct engine *e);

	/**
	 * Deletes a particle from the engine based on particle id.
	 *
	 * Afterwards, the particle id will point to a null entry in the partlist.
	 *
	 * Note, the next newly created particle will re-use this ID (assuming
	 * the engine_next_partid is used to determine the next id.)
	 */
	CAPI_FUNC(HRESULT) engine_del_particle(struct engine *e, int pid);

	// keep track of how frequently step is called, get average
	// steps per second, averaged over past 10 steps.
	CAPI_FUNC(FPTYPE) engine_steps_per_second();


	CAPI_FUNC(void) engine_dump();

	#define ENGINE_DUMP(msg) {std::cout<<msg<<std::endl; engine_dump();}

	CAPI_FUNC(FPTYPE) engine_kinetic_energy(struct engine *e);

	CAPI_FUNC(FPTYPE) engine_temperature(struct engine *e);

	#ifdef WITH_MPI
	CAPI_FUNC(int) engine_init_mpi(
		struct engine *e, 
		const FPTYPE *origin, 
		const FPTYPE *dim, 
		FPTYPE *L,
		FPTYPE cutoff, 
		unsigned int period, 
		int max_type, 
		unsigned int flags, 
		MPI_Comm comm,
		int rank
	);
	CAPI_FUNC(int) engine_exchange(struct engine *e);
	CAPI_FUNC(int) engine_exchange_async(struct engine *e);
	CAPI_FUNC(int) engine_exchange_async_run(struct engine *e);
	CAPI_FUNC(int) engine_exchange_incomming(struct engine *e);
	CAPI_FUNC(int) engine_exchange_rigid(struct engine *e);
	CAPI_FUNC(int) engine_exchange_rigid_async(struct engine *e);
	CAPI_FUNC(int) engine_exchange_rigid_async_run(struct engine *e);
	CAPI_FUNC(int) engine_exchange_rigid_wait(struct engine *e);
	CAPI_FUNC(int) engine_exchange_wait(struct engine *e);
	#endif

	#if defined(HAVE_CUDA)


	namespace cuda { 


		CAPI_FUNC(int) engine_nonbond_cuda(struct engine *e);
		CAPI_FUNC(int) engine_cuda_load(struct engine *e);
		CAPI_FUNC(int) engine_cuda_load_parts(struct engine *e);
		CAPI_FUNC(int) engine_cuda_unload_parts(struct engine *e);
		CAPI_FUNC(int) engine_cuda_load_pots(struct engine *e);
		CAPI_FUNC(int) engine_cuda_unload_pots(struct engine *e);
		CAPI_FUNC(int) engine_cuda_refresh_particles(struct engine *e);
		CAPI_FUNC(int) engine_cuda_allocate_particle_states(struct engine *e);
		CAPI_FUNC(int) engine_cuda_finalize_particle_states(struct engine *e);
		CAPI_FUNC(int) engine_cuda_refresh_particle_states(struct engine *e);
		CAPI_FUNC(int) engine_cuda_refresh_pots(struct engine *e);
		CAPI_FUNC(int) engine_cuda_load_fluxes(struct engine *e);
		CAPI_FUNC(int) engine_cuda_unload_fluxes(struct engine *e);
		CAPI_FUNC(int) engine_cuda_refresh_fluxes(struct engine *e);
		CAPI_FUNC(int) engine_cuda_boundary_conditions_refresh(struct engine *e);
		CAPI_FUNC(int) engine_cuda_queues_finalize(struct engine *e);
		CAPI_FUNC(int) engine_cuda_finalize(struct engine *e);
		CAPI_FUNC(int) engine_cuda_refresh(struct engine *e);
		CAPI_FUNC(int) engine_cuda_setthreads(struct engine *e, int id, int nr_threads);
		CAPI_FUNC(int) engine_cuda_setblocks(struct engine *e, int id, int nr_blocks);
		CAPI_FUNC(int) engine_cuda_setdevice(struct engine *e, int id);
		CAPI_FUNC(int) engine_cuda_setdevices(struct engine *e, int nr_devices, int *ids);
		CAPI_FUNC(int) engine_cuda_cleardevices(struct engine *e);
		CAPI_FUNC(int) engine_split_gpu(struct engine *e, int N, int flags);

		CAPI_FUNC(int) engine_toCUDA(struct engine *e);
		CAPI_FUNC(int) engine_fromCUDA(struct engine *e);

	}
	
	#endif

	#ifdef WITH_METIS
	CAPI_FUNC(int) engine_split_METIS(struct engine *e, int N, int flags);
	#endif

	FVector3 engine_origin();
	FVector3 engine_dimensions();
	FVector3 engine_center();

	/**
	 * Single static instance of the md engine per process.
	 *
	 * Even for MPI enabled, as each MPI process will initialize the engine with different comm and rank.
	 */
	// CAPI_DATA(engine) _Engine;
	extern CAPI_EXPORT engine _Engine;

	CPPAPI_FUNC(struct engine*) engine_get();

};


// inline definitions...

#include "tfParticle.h"


namespace TissueForge { 


	inline Particle *Particle::particle(int i) {
		return _Engine.s.partlist[this->parts[i]];
	};

	inline FVector3 Particle::global_position() {
		FVector3 position;
		space_getpos(&_Engine.s, this->id, position.data());
		return position;
	}

	inline void Particle::set_global_position(const FVector3& pos) {
		FPTYPE x[] = {pos.x(), pos.y(), pos.z()};
		space_setpos(&_Engine.s, this->id, x);
	}

	inline Particle *ParticleHandle::part() {
		return _Engine.s.partlist[this->id];
	};

	inline ParticleType *ParticleHandle::type() {
		return &_Engine.types[this->typeId];
	}

	inline Particle *Particle_FromId(int id) {
		return _Engine.s.partlist[id];
	}

	/**
	 * get the i'th particle that's a member of this type.
	 */
	inline Particle *ParticleType::particle(int i) {
		return _Engine.s.partlist[this->parts.parts[i]];
	}

};

#endif // _MDCORE_INCLUDE_TFENGINE_H_