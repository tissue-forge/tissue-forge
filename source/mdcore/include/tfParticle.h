/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
 * Coypright (c) 2017 Andy Somogyi (somogyie at indiana dot edu)
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

/**
 * @file tfParticle.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFPARTICLE_H_
#define _MDCORE_INCLUDE_TFPARTICLE_H_
#include "tf_platform.h"
#include "tf_fptype.h"
#include <state/tfStateVector.h>
#include <types/tf_types.h>
#include "tfSpace_cell.h"
#include "tfAngle.h"
#include "tfBond.h"
#include "tfDihedral.h"
#include "tfParticleList.h"
#include "tfParticleTypeList.h"
#include <set>
#include <vector>

/**
 * increment size of cluster particle list.
 */
#define CLUSTER_PARTLIST_INCR 50


namespace TissueForge { 


    namespace rendering {
        struct Style;
    }


    typedef enum ParticleTypeFlags {
        PARTICLE_TYPE_NONE          = 0,
        PARTICLE_TYPE_INERTIAL      = 1 << 0,
        PARTICLE_TYPE_DISSAPATIVE   = 1 << 1,
    } ParticleTypeFlags;

    typedef enum ParticleDynamics {
        PARTICLE_NEWTONIAN          = 0,
        PARTICLE_OVERDAMPED         = 1,
    } ParticleDynamics;

    /* particle flags */
    typedef enum ParticleFlags {
        PARTICLE_NONE          = 0,
        PARTICLE_GHOST         = 1 << 0,
        PARTICLE_CLUSTER       = 1 << 1,
        PARTICLE_BOUND         = 1 << 2,
        PARTICLE_FROZEN_X      = 1 << 3,
        PARTICLE_FROZEN_Y      = 1 << 4,
        PARTICLE_FROZEN_Z      = 1 << 5,
        PARTICLE_FROZEN        = PARTICLE_FROZEN_X | PARTICLE_FROZEN_Y | PARTICLE_FROZEN_Z,
        PARTICLE_LARGE         = 1 << 6,
    } ParticleFlags;


    /** ID of the last error. */
    // CAPI_DATA(int) particle_err;

    struct Cluster;
    struct ClusterParticleHandle;

    /**
     * The particle data structure.
     *
     * Instance vars for each particle.
     *
     * Note that the arrays for @c x, @c v and @c f are 4 entries long for
     * proper alignment.
     *
     * All particles are stored in a series of contiguous blocks of memory that are owned
     * by the space cells. Each space cell has a array of particle structs.
     * 
     * If you're building a model, you should probably instead be working with a 
     * ParticleHandle. 
     */
    struct CAPI_EXPORT Particle  {
        
        /**
         * Particle force
         *
         * ONLY the coherent part of the force should go here. We use multi-step
         * integrators, that need to separate the random and coherent forces.
         *
         * Force gets cleared each step, along with number density. 
         */
        union {
            TF_ALIGNED(FPTYPE, 16) f[4];
            TF_ALIGNED(FVector3, 16) force;
            
            struct {
                FPTYPE __dummy0[3];
                FPTYPE number_density;
            };
        };
        
        /**
         * Initial particle force
         *
         * At each step, the total force acting on a particle is reset to this settable value. 
         */
        union {
            TF_ALIGNED(FVector3, 16) force_i;
        };


        /** Particle velocity */
        union {
            TF_ALIGNED(FPTYPE, 16) v[4];
            TF_ALIGNED(FVector3, 16) velocity;
            
            struct {
                FPTYPE __dummy1[3];
                FPTYPE inv_number_density;
            };
        };

        
        /** Particle position */
        union {
            TF_ALIGNED(FPTYPE, 16) x[4];
            TF_ALIGNED(FVector3, 16) position;

            struct {
                FPTYPE __dummy2[3];
                uint32_t creation_time;
            };
        };
        

        /** Random force. */
        union {
            TF_ALIGNED(FVector3, 16) persistent_force;
        };

        /** Inverse mass */
        FPTYPE imass;

        /** Particle radius */
        FPTYPE radius;

        /** Particle mass */
        FPTYPE mass;

        /** Individual particle charge, if needed */
        FPTYPE q;

        // runge-kutta k intermediates.
        FVector3 p0;
        FVector3 v0;
        FVector3 xk[4];
        FVector3 vk[4];

        /**
         * Particle id, virtual id
         * TODO: not sure what virtual id is...
         */
        int id, vid;

        /** Particle type id */
        int16_t typeId;

        /** Cluster id of this part */
        int32_t clusterId;

        /** Particle flags */
        uint16_t flags;

        /**
         * pointer to the handle. 
         * 
         * Particle data gets moved around between cells, 
         * and the handle provides a safe way to retrieve them.
         */
        struct ParticleHandle *_handle;

        /**
         * @brief Get a handle for this particle. 
         */
        struct ParticleHandle *handle();

        /**
         * list of particle ids that belong to this particle, if it is a cluster.
         */
        int32_t *parts;

        /** number of particle ids that belong to this particle, if it is a cluster. */
        uint16_t nr_parts;

        /** size of particle ids that belong to this particle, if it is a cluster. */
        uint16_t size_parts;

        /**
         * add a particle (id) to this type
         */
        HRESULT addpart(int32_t uid);

        /**
         * removes a particle from this cluster. Sets the particle cluster id
         * to -1, and removes if from this cluster's list.
         */
        HRESULT removepart(int32_t uid);

        /** Get the ith particle, if this particle is a cluster. */
        Particle *particle(int i);

        /**
         * Style pointer, set at object construction time. 
         * Can be set by the user after construction. 
         * The base particle type has a default style.
         */
        rendering::Style *style;

        /**
         * optional pointer to state vector
         * 
         * Set at construction time when species are present in a simulation.
         */
        struct state::StateVector *state_vector;

        /**
         * @brief Get the global position
         */
        FVector3 global_position();

        /**
         * @brief Set the global position
         */
        void set_global_position(const FVector3& pos);
        
        /**
         * performs a self-verify, in debug mode raises assertion if not valid
         */
        bool verify();

        
        /**
         * @brief Cast to a cluster type. Limits casting to cluster by type. 
         * 
         * @return Cluster* 
         */
        operator Cluster*();
        
        Particle();

        /**
         * @brief Get a JSON string representation
         */
        std::string toString();

        /**
         * @brief Create from a JSON string representation. 
         * 
         * The returned particle is not automatically registered with the engine. 
         * 
         * To properly register a particle from a string, pass the string to the 
         * particle constructor of the appropriate particle type or cluster. 
         * 
         * @param str constructor string, as returned by toString
         * @return unregistered particle
         */
        static Particle *fromString(const std::string &str);
    };

    /**
     * iterates over all parts and does a verify
     */
    HRESULT Particle_Verify();

    #ifndef NDEBUG
    #define VERIFY_PARTICLES() Particle_Verify()
    #else
    #define VERIFY_PARTICLES()
    #endif


    /**
     * @brief A handle to a particle. 
     *
     * The engine allocates particle memory in blocks, and particle
     * values get moved around all the time, so their addresses change.
     *
     * The partlist is always ordered by id, i.e. partlist[id]  always
     * points to the same particle, even though that particle may move
     * from cell to cell.
     * 
     * This is a safe way to work with a particle. 
     */
    struct CAPI_EXPORT ParticleHandle {
        /** Particle id */
        int id;

        /**
         * @brief Gets the actual particle of this handle. 
         * 
         * @return particle, if available
         */
        Particle *part();

        /**
         * @brief Gets the actual particle of this handle. 
         * 
         * Alias for consistency with other objects. 
         */
        Particle *get() { return part(); }

        /**
         * @brief Gets the particle type of this handle. 
         */
        ParticleType *type();

        ParticleHandle() : id(0) {}
        ParticleHandle(const int &id) : id(id) {}

        virtual std::string str() const;

        virtual ParticleHandle* fission();

        /**
         * @brief Splits a single particle into two and returns the new particle. 
         * 
         * @return new particle
         */
        virtual ParticleHandle* split();

        /**
         * @brief Splits a single particle into two along a direction. 
         * 
         * @param direction direction along which the particle is split. 
         * @return new particle
        */
        virtual ParticleHandle* split(const FVector3& direction);

        /**
         * @brief Destroys the particle and removes it from inventory. 
         * 
         * Subsequent references to a destroyed particle result in an error. 
         */
        HRESULT destroy();

        /**
         * @brief Calculates the particle's coordinates in spherical coordinates. 
         * 
         * By default, calculations are made with respect to the center of the universe. 
         * 
         * @param particle a particle to use as the origin, optional
         * @param origin a point to use as the origin, optional
         */
        FVector3 sphericalPosition(Particle *particle=NULL, FVector3 *origin=NULL);

        /**
         * @brief Computes the relative position with respect to an origin while 
         * optionally account for boundary conditions. 
         * 
         * @param origin origin
         * @param comp_bc flag to compensate for boundary conditions; default true
         */
        FVector3 relativePosition(const FVector3 &origin, const bool &comp_bc=true);

        /**
         * @brief Computes the virial tensor. Optionally pass a distance to include a neighborhood. 
         * 
         * @param radius A distance to define a neighborhood, optional
         */
        virtual FMatrix3 virial(FPTYPE *radius=NULL);

        /**
         * @brief Dynamically changes the *type* of an object. We can change the type of a 
         * ParticleType-derived object to anyther pre-existing ParticleType-derived 
         * type. What this means is that if we have an object of say type 
         * *A*, we can change it to another type, say *B*, and and all of the forces 
         * and processes that acted on objects of type A stip and the forces and 
         * processes defined for type B now take over. 
         * 
         * @param type new particle type
         */
        HRESULT become(ParticleType *type);

        /**
         * @brief Gets a list of nearby particles. 
         * 
         * @param distance search distance
         * @param types list of particle types to search by
         */
        ParticleList neighbors(const FPTYPE &distance, const TissueForge::ParticleTypeList &types);

        /**
         * @brief Gets a list of nearby particles. 
         * 
         * @param distance search distance
         * @param types list of particle types to search by
         */
        ParticleList neighbors(const FPTYPE &distance, const std::vector<ParticleType> &types);

        /**
         * @brief Gets a list of nearby particles of all types. 
         * 
         * @param distance search distance
         */
        ParticleList neighbors(const FPTYPE &distance);

        /**
         * @brief Gets a list of nearby particles within the global cutoff distance. 
         * 
         * @param types list of particle types to search by
         */
        ParticleList neighbors(const TissueForge::ParticleTypeList &types);

        /**
         * @brief Gets a list of nearby particles within the global cutoff distance. 
         * 
         * @param types list of particle types to search by
         */
        ParticleList neighbors(const std::vector<ParticleType> &types);

        /**
         * @brief Gets a list of nearby particles ids. 
         * 
         * @param distance search distance
         * @param types list of particle types to search by
         */
        std::vector<int32_t> neighborIds(const FPTYPE &distance, const TissueForge::ParticleTypeList &types);

        /**
         * @brief Gets a list of nearby particles ids. 
         * 
         * @param distance search distance
         * @param types list of particle types to search by
         */
        std::vector<int32_t> neighborIds(const FPTYPE &distance, const std::vector<ParticleType> &types);

        /**
         * @brief Gets a list of nearby particles ids of all types. 
         * 
         * @param distance search distance
         */
        std::vector<int32_t> neighborIds(const FPTYPE &distance);

        /**
         * @brief Gets a list of nearby particles ids within the global cutoff distance. 
         * 
         * @param types list of particle types to search by
         */
        std::vector<int32_t> neighborIds(const TissueForge::ParticleTypeList &types);

        /**
         * @brief Gets a list of nearby particles ids within the global cutoff distance. 
         * 
         * @param types list of particle types to search by
         */
        std::vector<int32_t> neighborIds(const std::vector<ParticleType> &types);

        /**
         * @brief Gets a list of all bonded neighbors. 
         */
        ParticleList getBondedNeighbors();

        /**
         * @brief Gets a list of all bonded neighbor ids. 
         */
        std::vector<int32_t> getBondedNeighborIds();

        /**
         * @brief Calculates the distance to another particle
         * 
         * @param _other another particle. 
         */
        FPTYPE distance(ParticleHandle *_other);

        /** Get the bonds attached to the particle. */
        std::vector<struct TissueForge::BondHandle> getBonds();

        /** Get the angles attached to the particle. */
        std::vector<struct TissueForge::AngleHandle> getAngles();

        /** Get the dihedrals attached to the particle. */
        std::vector<struct TissueForge::DihedralHandle> getDihedrals();

        /** Particle charge */
        FPTYPE getCharge();

        /** Set the particle charge */
        void setCharge(const FPTYPE &charge);

        /** Particle mass */
        FPTYPE getMass();
        
        /** Set the particle mass */
        void setMass(const FPTYPE &mass);
        
        /** Particle frozen state */
        bool getFrozen();
        
        /** Set the particle frozen state */
        void setFrozen(const bool frozen);
        
        /** Particle frozen state along the x-direction */
        bool getFrozenX();
        
        /** Set the particle frozen state along the x-direction */
        void setFrozenX(const bool frozen);
        
        /** Particle frozen state along the y-direction */
        bool getFrozenY();
        
        /** Set the particle frozen state along the y-direction */
        void setFrozenY(const bool frozen);
        
        /** Particle frozen state along the z-direction */
        bool getFrozenZ();
        
        /** Set the particle frozen state along the z-direction */
        void setFrozenZ(const bool frozen);
        
        /** Particle style */
        rendering::Style *getStyle();
        
        /** Set the prticle style */
        void setStyle(rendering::Style *style);
        
        /** Particle age */
        FPTYPE getAge();
        
        /** Particle radius */
        FPTYPE getRadius();
        
        /** Set the particle radius */
        void setRadius(const FPTYPE &radius);

        /** Particle volume */
        FPTYPE getVolume();
        
        /** Particle name */
        std::string getName();
        
        /** Particle second name */
        std::string getName2();
        
        /** Particle position */
        FVector3 getPosition();
        
        /** Set the particle position */
        void setPosition(FVector3 position);
        
        /** Particle velocity */
        FVector3 &getVelocity();
        
        /** Set the particle velocity */
        void setVelocity(FVector3 velocity);
        
        /** Particle force */
        FVector3 getForce();
        
        /** Particle initial force */
        FVector3 &getForceInit();
        
        /** Set the particle initial force */
        void setForceInit(FVector3 force);
        
        /** Particle id */
        int getId();
        
        /** Particle type id */
        int16_t getTypeId();
        
        /** Particle cluster id */
        int32_t getClusterId();
        
        /** Particle flags */
        uint16_t getFlags();
        
        /** Particle species, if any */
        state::StateVector *getSpecies();

        /**
         * Limits casting to cluster by type
         */
        operator ClusterParticleHandle*();
    };

    /**
     * @brief Structure containing information on each particle type.
     *
     * This is only a definition for a *type* of particle, and not an 
     * actual particle with position, velocity, etc. However, particles 
     * of this *type* can be created with one of these. 
     */
    struct CAPI_EXPORT ParticleType {

        static const int MAX_NAME = 64;

        /** ID of this type */
        int16_t id;

        /**
         *  type flags
         */
        uint32_t type_flags;

        /**
         * particle flags, the type initializer sets these, and
         * all new particle instances get a copy of these.
         */
        uint16_t particle_flags;

        /**
         * @brief Default mass of particles
         */
        FPTYPE mass;
        
        FPTYPE imass;
        
        /**
         * @brief Default charge of particles
         */
        FPTYPE charge;

        /**
         * @brief Default radius of particles
         */
        FPTYPE radius;

        /**
         * @brief Kinetic energy of all particles of this type. 
         */
        FPTYPE kinetic_energy;

        /**
         * @brief Potential energy of all particles of this type. 
         */
        FPTYPE potential_energy;

        /**
         * @brief Target energy of all particles of this type. 
         */
        FPTYPE target_energy;

        /**
         * @brief Default minimum radius of this type. 
         * 
         * If a split event occurs, resulting particles will have a radius 
         * at least as great as this value. 
         */
        FPTYPE minimum_radius;

        /** Nonbonded interaction parameters. */
        FPTYPE eps, rmin;

        /**
         * @brief Default dynamics of particles of this type. 
         */
        unsigned char dynamics;

        /**
         * @brief Name of this particle type.
         */
        char name[MAX_NAME];
        
        char name2[MAX_NAME];

        /**
         * @brief list of particles that belong to this type.
         */
        TissueForge::ParticleList parts;

        /**
         * @brief list of particle types that belong to this type.
         */
        TissueForge::ParticleTypeList types;

        /**
         * @brief style pointer, optional.
         */
        rendering::Style *style;

        /**
         * @brief optional pointer to species list. This is the metadata for the species
         */
        struct state::SpeciesList *species = 0;

        /**
         * add a particle (id) to this type
         */
        HRESULT addpart(int32_t id);

        /**
         * remove a particle id from this type
         */
        HRESULT del_part(int32_t id);

        /**
         * @brief get the i'th particle that's a member of this type.
         * 
         * @param i index of particle to get
         */
        TissueForge::Particle *particle(int i);

        /**
         * @brief Get all current particle type ids, excluding clusters
         */
        static std::set<short int> particleTypeIds();

        /** Test whether the type is a cluster type */
        bool isCluster();

        /**
         * Limits casting to cluster by type
         */
        operator TissueForge::ClusterParticleType*();

        /**
         * @brief Particle constructor. 
         * 
         * Automatically updates when running on a CUDA device. 
         * 
         * @param position position of new particle, optional
         * @param velocity velocity of new particle, optional
         * @param clusterId id of parent cluster, optional
         * @return new particle
         */
        TissueForge::ParticleHandle *operator()(
            FVector3 *position=NULL,
            FVector3 *velocity=NULL,
            int *clusterId=NULL
        );
        
        /**
         * @brief Particle constructor. 
         * 
         * Automatically updates when running on a CUDA device. 
         * 
         * @param str JSON string
         * @param clusterId id of parent cluster, optional
         * @return new particle
         */
        TissueForge::ParticleHandle *operator()(const std::string &str, int *clusterId=NULL);

        /**
         * @brief Particle factory constructor, for making lots of particles quickly. 
         * 
         * At minimum, arguments must specify the number of particles to create, whether 
         * specified explicitly or through one or more vector arguments.
         * 
         * @param nr_parts number of particles to create, optional
         * @param positions initial particle positions, optional
         * @param velocities initial particle velocities, optional
         * @param clusterIds parent cluster ids, optional
         * @return new particle ids
         */
        std::vector<int> factory(
            unsigned int nr_parts=0, 
            std::vector<FVector3> *positions=NULL, 
            std::vector<FVector3> *velocities=NULL, 
            std::vector<int> *clusterIds=NULL
        );

        /**
         * @brief Particle type constructor. 
         * 
         * New type is constructed from the definition of the calling type. 
         * 
         * @param _name name of the new type
         * @return new particle type
         */
        ParticleType* newType(const char *_name);

        /** Test whether the type has an id */
        bool has(const int32_t &pid);

        /** Test whether the type has a particle */
        bool has(ParticleHandle *part);

        /**
         * @brief Registers a type with the engine.
         * 
         * Note that this occurs automatically, unless noReg==true in constructor.  
         */
        virtual HRESULT registerType();

        /**
         * @brief A callback for when a type is registered
         */
        virtual void on_register() {}

        /**
         * @brief Tests whether this type is registered
         * 
         * @return true if registered
         */
        bool isRegistered();

        /**
         * @brief Get the type engine instance
         */
        virtual ParticleType *get();

        ParticleType(const bool &noReg=false);
        virtual ~ParticleType() {}

        virtual std::string str() const;

        /** Type volume */
        FPTYPE getVolume();

        /** Type frozen state */
        bool getFrozen();
        
        /** Set the type frozen state */
        void setFrozen(const bool &frozen);
        
        /** Type frozen state along the x-direction */
        bool getFrozenX();
        
        /** Set the type frozen state along the x-direction */
        void setFrozenX(const bool &frozen);
        
        /** Type frozen state along the y-direction */
        bool getFrozenY();
        
        /** Set the type frozen state along the y-direction */
        void setFrozenY(const bool &frozen);
        
        /** Type frozen state along the z-direction */
        bool getFrozenZ();
        
        /** Set the type frozen state along the z-direction */
        void setFrozenZ(const bool &frozen);
        
        /** Type temperature, an ensemble property */
        FPTYPE getTemperature();
        
        /** Type target temperature, an ensemble property */
        FPTYPE getTargetTemperature();
        
        /** Set the type target temperature, an ensemble property */
        void setTargetTemperature(const FPTYPE &temperature);

        /**
         * @brief Get all particles of this type. 
         */
        TissueForge::ParticleList &items();

        /** number of particles that belong to this type. */
        uint16_t getNumParts();

        /** list of particle ids that belong to this type. */
        std::vector<int32_t> getPartIds();

        /**
         * @brief Get a JSON string representation
         */
        std::string toString();

        /**
         * @brief Create from a JSON string representation. 
         * 
         * The returned type is automatically registered with the engine. 
         * 
         * @param str a string, as returned by ``toString``
         */
        static ParticleType *fromString(const std::string &str);
    };

    CAPI_FUNC(ParticleType*) Particle_GetType();

    CAPI_FUNC(ParticleType*) Cluster_GetType();

    /**
     * Creates a new ParticleType for the given particle data pointer.
     *
     * This creates a matching python type for an existing particle data,
     * and is usually called when new types are created from C.
     */
    ParticleType *ParticleType_ForEngine(
        struct engine *e, 
        FPTYPE mass, 
        FPTYPE charge,
        const char *name, 
        const char *name2
    );

    /**
     * Creates and initializes a new particle type, adds it to the
     * global engine
     *
     * creates both a new type, and a new data entry in the engine.
     */
    ParticleType* ParticleType_New(const char *_name);

    /**
     * @brief Get a registered particle type by type name
     * 
     * @param name name of particle type
     * @return particle type, if found
     */
    CAPI_FUNC(ParticleType*) ParticleType_FindFromName(const char* name);

    /**
     * checks if a python object is a particle, and returns the
     * corresponding particle pointer, NULL otherwise
     */
    CAPI_FUNC(Particle*) Particle_Get(ParticleHandle *pypart);


    /**
     * simple fission,
     *
     * divides a particle into two, and creates a new daughter particle in the
     * universe.
     *
     * Vector of numbers indicate how to split the attached chemical cargo.
     */
    CAPI_FUNC(ParticleHandle*) Particle_FissionSimple(
        Particle *part,
        ParticleType *a, 
        ParticleType *b,
        int nPartitionRatios, 
        FPTYPE *partitionRations
    );

    CAPI_FUNC(ParticleHandle*) Particle_New(
        ParticleType *type, 
        FVector3 *positn=NULL,
        FVector3 *velocity=NULL,
        int *clusterId=NULL
    );

    std::vector<int> Particles_New(
        std::vector<ParticleType*> types, 
        std::vector<FVector3> *positions=NULL, 
        std::vector<FVector3> *velocities=NULL, 
        std::vector<int> *clusterIds=NULL
    );

    std::vector<int> Particles_New(
        ParticleType *type, 
        unsigned int nr_parts=0, 
        std::vector<FVector3> *positions=NULL, 
        std::vector<FVector3> *velocities=NULL, 
        std::vector<int> *clusterIds=NULL
    );

    /**
     * Change the type of one particle to another.
     *
     * removes the particle from it's current type's list of objects,
     * and adds it to the new types list.
     *
     * changes the type pointer in the C Particle, and also changes
     * the type pointer in the Python PyParticle handle.
     */
    CAPI_FUNC(HRESULT) Particle_Become(Particle *part, ParticleType *type);

    /**
     * The the particle type type
     */
    CAPI_DATA(unsigned int) *Particle_Colors;

    // Returns 1 if a type has been registered, otherwise 0
    CAPI_FUNC(bool) ParticleType_checkRegistered(ParticleType *type);

    /**
     * mandatory internal function to initalize the particle and particle types
     *
     * sets the engine.types[0] particle.
     *
     * The engine.types array is assumed to be allocated, but not initialized.
     */
    HRESULT _Particle_init();

    inline bool operator< (const TissueForge::ParticleHandle& lhs, const TissueForge::ParticleHandle& rhs) { return lhs.id < rhs.id; }
    inline bool operator> (const TissueForge::ParticleHandle& lhs, const TissueForge::ParticleHandle& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::ParticleHandle& lhs, const TissueForge::ParticleHandle& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::ParticleHandle& lhs, const TissueForge::ParticleHandle& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::ParticleHandle& lhs, const TissueForge::ParticleHandle& rhs) { return lhs.id == rhs.id; }
    inline bool operator!=(const TissueForge::ParticleHandle& lhs, const TissueForge::ParticleHandle& rhs) { return !(lhs == rhs); }

    inline bool operator< (const TissueForge::ParticleType& lhs, const TissueForge::ParticleType& rhs) { return lhs.id < rhs.id; }
    inline bool operator> (const TissueForge::ParticleType& lhs, const TissueForge::ParticleType& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::ParticleType& lhs, const TissueForge::ParticleType& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::ParticleType& lhs, const TissueForge::ParticleType& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::ParticleType& lhs, const TissueForge::ParticleType& rhs) { return lhs.id == rhs.id; }
    inline bool operator!=(const TissueForge::ParticleType& lhs, const TissueForge::ParticleType& rhs) { return !(lhs == rhs); }


};


inline std::ostream &operator<<(std::ostream& os, const TissueForge::ParticleHandle &p)
{
    os << p.str().c_str();
    return os;
}

inline std::ostream &operator<<(std::ostream& os, const TissueForge::ParticleType &p)
{
    os << p.str().c_str();
    return os;
}

#endif // _MDCORE_INCLUDE_TFPARTICLE_H_