/*******************************************************************************
 * This file is part of mdcore.
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
 * @file tfBoundaryConditions.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFBOUNDARYCONDITIONS_H_
#define _MDCORE_INCLUDE_TFBOUNDARYCONDITIONS_H_

#include <mdcore_config.h>
#include <types/tf_types.h>
#include <io/tf_io.h>

#include <unordered_map>


namespace TissueForge {


    enum BoundaryConditionKind : unsigned int {
        BOUNDARY_VELOCITY       = 1 << 0,
        BOUNDARY_PERIODIC       = 1 << 1,
        BOUNDARY_FREESLIP       = 1 << 2,
        BOUNDARY_POTENTIAL      = 1 << 3,
        BOUNDARY_NO_SLIP        = 1 << 4, // really just velocity with zero velocity
        BOUNDARY_RESETTING      = 1 << 5, // reset the chemical cargo when particles cross boundaries. 
        BOUNDARY_ACTIVE         = BOUNDARY_FREESLIP | BOUNDARY_VELOCITY | BOUNDARY_POTENTIAL
    };

    struct ParticleType;
    struct Potential;

    /**
     * @brief A condition on a boundary of the universe. 
     * 
     */
    struct CAPI_EXPORT BoundaryCondition {
        BoundaryConditionKind kind;

        // id of this boundary, id's go from 0 to 6 (top, bottom, etc..)
        int id;

        /**
         * @brief the velocity on the boundary
         */
        FVector3 velocity;

        /** 
         * @brief restoring percent. 
         * 
         * When objects hit this boundary, they get reflected back at `restore` percent, 
         * so if restore is 0.5, and object hitting the boundary at 3 length / time 
         * recoils with a velocity of 1.5 lengths / time. 
         */
        FPTYPE restore;

        /**
         * @brief name of the boundary
         */
        const char* name;
        
        /**
         * @brief vector normal to the boundary
         */
        FVector3 normal;

        /**
         * pointer to offset in main array allocated in BoundaryConditions.
         */
        struct Potential **potenntials;

        // many potentials act on the sum of both particle radii, so this
        // paramter makes it looks like the wall has a sheet of particles of
        // radius.
        FPTYPE radius;

        /**
         * sets the potential for the given particle type.
         */
        void set_potential(struct ParticleType *ptype, struct Potential *pot);

        std::string kindStr() const;
        std::string str(bool show_name) const;

        unsigned init(const unsigned &kind);
        unsigned init(const FVector3 &velocity, const FPTYPE *restore=NULL);
        unsigned init(
            const std::unordered_map<std::string, unsigned int> vals, 
            const std::unordered_map<std::string, FVector3> vels, 
            const std::unordered_map<std::string, FPTYPE> restores
        );
    };

    /**
     * @brief The BoundaryConditions class serves as a container for the six 
     * instances of the :class:`BoundaryCondition` object
     * 
     */
    struct CAPI_EXPORT BoundaryConditions {

        /**
         * @brief The top boundary
         */
        BoundaryCondition top;

        /**
         * @brief The bottom boundary
         */
        BoundaryCondition bottom;

        /**
         * @brief The left boundary
         */
        BoundaryCondition left;

        /**
         * @brief The right boundary
         */
        BoundaryCondition right;

        /**
         * @brief The front boundary
         */
        BoundaryCondition front;

        /**
         * @brief The back boundary
         */
        BoundaryCondition back;

        // pointer to big array of potentials, 6 * max types.
        // each boundary condition has a pointer that's an offset
        // into this array, so allocate and free in single block.
        // allocated in BoundaryConditions_Init.
        struct Potential **potenntials;

        BoundaryConditions() {}
        BoundaryConditions(int *cells);
        BoundaryConditions(int *cells, const int &value);
        BoundaryConditions(
            int *cells, 
            const std::unordered_map<std::string, unsigned int> vals, 
            const std::unordered_map<std::string, FVector3> vels, 
            const std::unordered_map<std::string, FPTYPE> restores
        );

        /**
         * @brief sets a potential for ALL boundary conditions and the given potential.
         * 
         * @param ptype particle type
         * @param pot potential
         */
        void set_potential(struct ParticleType *ptype, struct Potential *pot);

        /**
         * bitmask of periodic boundary conditions
         */
        uint32_t periodic;

        static unsigned boundaryKindFromString(const std::string &s);
        static unsigned boundaryKindFromStrings(const std::vector<std::string> &kinds);

        std::string str();

        /**
         * @brief Get a JSON string representation
         * 
         * @return std::string 
         */
        std::string toString();

        /**
         * @brief Create from a JSON string representation. 
         * 
         * @param str 
         * @return BoundaryConditions* 
         */
        static BoundaryConditions *fromString(const std::string &str);

    private:

        HRESULT _initIni();
        HRESULT _initFin(int *cells);

        // processes directional initialization inputs
        void _initDirections(const std::unordered_map<std::string, unsigned int> vals);

        // processes sides initialization inputs
        void _initSides(
            const std::unordered_map<std::string, unsigned int> vals, 
            const std::unordered_map<std::string, FVector3> vels, 
            const std::unordered_map<std::string, FPTYPE> restores
        );
    };

    struct CAPI_EXPORT BoundaryConditionsArgsContainer {
        int *bcValue;
        std::unordered_map<std::string, unsigned int> *bcVals;
        std::unordered_map<std::string, FVector3> *bcVels;
        std::unordered_map<std::string, FPTYPE> *bcRestores;

        void setValueAll(const int &_bcValue);
        void setValue(const std::string &name, const unsigned int &value);
        void setVelocity(const std::string &name, const FVector3 &velocity);
        void setRestore(const std::string &name, const FPTYPE restore);

        BoundaryConditions *create(int *cells);

        BoundaryConditionsArgsContainer(
            int *_bcValue=NULL, 
            std::unordered_map<std::string, unsigned int> *_bcVals=NULL, 
            std::unordered_map<std::string, FVector3> *_bcVels=NULL, 
            std::unordered_map<std::string, FPTYPE> *_bcRestores=NULL
        );

    private:
        void switchType(const bool &allSides);
    };

    /**
     * a particle moved from one cell to another, this checks if its a periodic
     * crossing, and adjusts any particle state values if the boundaries say so.
     */
    void apply_boundary_particle_crossing(
        struct Particle *p, 
        const int *delta,
        const struct space_cell *source_cell, 
        const struct space_cell *dest_cell
    );


};

#endif // _MDCORE_INCLUDE_TFBOUNDARYCONDITIONS_H_