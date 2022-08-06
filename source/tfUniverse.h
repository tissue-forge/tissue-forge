/*******************************************************************************
 * This file is part of Tissue Forge.
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
 * @file tfUniverse.h
 * 
 */

#ifndef _SOURCE_TFUNIVERSE_H_
#define _SOURCE_TFUNIVERSE_H_

#include "TissueForge_private.h"
#ifdef TF_FPTYPE_SINGLE
#include <mdcore_single.h>
#else
#include <mdcore_double.h>
#endif
#include "io/tf_io.h"
#include <unordered_map>
#include "event/tfEventList.h"


namespace TissueForge { 


    /**
     * @brief The universe is a top level singleton object, and is automatically
     * initialized when the simulator loads. The universe is a representation of the
     * physical universe that we are simulating, and is the repository for all
     * physical object representations.
     * 
     * All properties and methods on the universe are static, and you never actually
     * instantiate a universe.
     * 
     * Universe has a variety of properties such as boundary conditions, and stores
     * all the physical objects such as particles, bonds, potentials, etc.
     */
    struct CAPI_EXPORT Universe  {

        enum Flags {
            RUNNING = 1 << 0,

            SHOW_PERF_STATS = 1 << 1,

            // in ipython message loop, monitor console
            IPYTHON_MSGLOOP = 1 << 2,

            // standard polling message loop
            POLLING_MSGLOOP = 1 << 3,
        };

        /**
         * @brief Gets the dimensions of the universe
         * 
         * @return FVector3 
         */
        static FVector3 dim();

        bool isRunning;

        event::EventBaseList *events;

        // name of the model / script, usually picked up from command line;
        std::string name;

        /**
         * @brief Get the name of the model / script
         */
        static std::string getName();
        
        /**
         * @brief Computes the virial tensor for the either the entire simulation 
         * domain, or a specific local virial tensor at a location and 
         * radius. Optionally can accept a list of particle types to restrict the 
         * virial calculation for specify types.
         * 
         * @param origin An optional length-3 array for the origin. Defaults to the center of the simulation domain if not given.
         * @param radius An optional number specifying the size of the region to compute the virial tensor for. Defaults to the entire simulation domain.
         * @param types An optional list of :class:`Particle` types to include in the calculation. Defaults to every particle type.
         */
        static FMatrix3 *virial(FVector3 *origin=NULL, FloatP_t *radius=NULL, std::vector<ParticleType*> *types=NULL);

        /** Center of the universe */
        static FVector3 getCenter();

        /**
         * @brief Performs a single time step ``dt`` of the universe if no arguments are 
         * given. Optionally runs until ``until``, and can use a different timestep 
         * of ``dt``.
         * 
         * @param until runs the timestep for this length of time, optional.
         * @param dt overrides the existing time step, and uses this value for time stepping; currently not supported.
         */
        static HRESULT step(const FloatP_t &until=0, const FloatP_t &dt=0);

        /**
         * @brief Stops the universe time evolution. This essentially freezes the universe, 
         * everything remains the same, except time no longer moves forward.
         */
        static HRESULT stop();

        /**
         * @brief Starts the universe time evolution, and advanced the universe forward by 
         * timesteps in ``dt``. All methods to build and manipulate universe objects 
         * are valid whether the universe time evolution is running or stopped.
         */
        static HRESULT start();

        static HRESULT reset();

        static Universe* get();

        /**
         * @brief Gets all particles in the universe
         */
        static ParticleList *particles();

        /**
         * @brief Reset all species in all particles
         * 
         */
        static void resetSpecies();

        /**
         * @brief Gets a three-dimesional array of particle lists, of all the particles in the system. 
         * 
         * @param shape shape of grid
         */
        static std::vector<std::vector<std::vector<ParticleList*> > > grid(iVector3 shape);

        /**
         * @brief Get all bonds in the universe
         */
        static std::vector<BondHandle> bonds();

        /**
         * @brief Get all angles in the universe
         */
        static std::vector<AngleHandle> angles();

        /**
         * @brief Get all dihedrals in the universe
         */
        static std::vector<DihedralHandle> dihedrals();

        /**
         * @brief Get the universe temperature. 
         * 
         * The universe can be run with, or without a thermostat. With a thermostat, 
         * getting / setting the temperature changes the temperature that the thermostat 
         * will try to keep the universe at. When the universe is run without a 
         * thermostat, reading the temperature returns the computed universe temp, but 
         * attempting to set the temperature yields an error. 
         */
        static FloatP_t getTemperature();

        /**
         * @brief Get the current time
         */
        static FloatP_t getTime();

        /**
         * @brief Get the period of a time step
         */
        static FloatP_t getDt();
        static event::EventList *getEventList();

        /** Universe boundary conditions */
        static BoundaryConditions *getBoundaryConditions();

        /**
         * @brief Get the current system kinetic energy
         */
        static FloatP_t getKineticEnergy();

        /**
         * @brief Get the current number of registered particle types
         */
        static int getNumTypes();

        /**
         * @brief Get the global interaction cutoff distance
         */
        static FloatP_t getCutoff();
    };

    /**
     *
     * @brief Initialize an #engine with the given data.
     *
     * The number of spatial cells in each cartesion dimension is floor( dim[i] / L[i] ), or
     * the physical size of the space in that dimension divided by the minimum size size of
     * each cell.
     *
     * @param e The #engine to initialize.
     * @param dim An array of three doubles containing the size of the space.
     *
     * @param L The minimum spatial cell edge length in each dimension.
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

    struct CAPI_EXPORT UniverseConfig {
        /** Dimensions of the universe */
        FVector3 dim;

        /** Discretization of the universe */
        iVector3 spaceGridSize;

        /** Global cutoff */
        FloatP_t cutoff;
        uint32_t flags;
        uint32_t maxTypes;

        /** Simulation time step, in units of simulation time */
        FloatP_t dt;

        /** Starting step of simulation */
        long start_step;
        FloatP_t temp;
        int nParticles;

        /** Number of simulation threads */
        int threads;

        /** Type of integrator */
        EngineIntegrator integrator;
        
        // pointer to boundary conditions ctor data
        // these objects are parsed initializing the engine.
        BoundaryConditionsArgsContainer *boundaryConditionsPtr;

        FloatP_t max_distance;
        
        // bitmask of timers to show in performance counter output.
        uint32_t timers_mask;
        
        long timer_output_period;
        
        UniverseConfig();
        
        // just set the object, borow a pointer to python handle
        void setBoundaryConditions(BoundaryConditionsArgsContainer *_bcArgs) {
            boundaryConditionsPtr = _bcArgs;
        }
        
        ~UniverseConfig() {
            if(boundaryConditionsPtr) {
                delete boundaryConditionsPtr;
                boundaryConditionsPtr = 0;
            }
        }
    };


    /**
     * runs the universe a pre-determined period of time, until.
     * can use micro time steps of 'dt' which override the
     * saved universe dt.
     *
     * if until is 0, it is ignored and the universe.dt is used.
     * if dt is 0, it is ignored, and the universe.dt is used as
     * a single time step.
     */
    CAPI_FUNC(HRESULT) Universe_Step(FloatP_t until, FloatP_t dt);

    /**
     * get a flag value
     */
    CAPI_FUNC(int) Universe_Flag(Universe::Flags flag);

    /**
     * sets / clears a flag value
     */
    CAPI_FUNC(HRESULT) Universe_SetFlag(Universe::Flags flag, int value);



    /**
     * The single global instance of the universe
     */
    extern CAPI_EXPORT Universe _Universe;

    /**
     * Universe instance accessor
     * 
     */
    CAPI_FUNC(Universe*) getUniverse();


    namespace io { 

        template <>
        HRESULT toFile(const Universe &dataElement, const MetaData &metaData, IOElement *fileElement);

        template <>
        HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Universe *dataElement);

    }

};

#endif // _SOURCE_TFUNIVERSE_H_