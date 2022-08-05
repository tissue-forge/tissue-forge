/*******************************************************************************
 * This file is part of mdcore.
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
 * @file tfParticleList.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFPARTICLELIST_H_
#define _MDCORE_INCLUDE_TFPARTICLELIST_H_

#include <mdcore_config.h>
#include <io/tf_io.h>
#include <types/tf_types.h>

#include <vector>


namespace TissueForge { 


    enum ParticleListFlags {
        // list owns the data the ParticleList::parts
        PARTICLELIST_OWNDATA = 1 << 0,
        
        // list supports insertion / deletion
        PARTICLELIST_MUTABLE = 1 << 1,
        
        // list owns it's own data, it was allocated.
        PARTICLELIST_OWNSELF = 1 << 2,
    };

    struct ParticleHandle;

    /**
     * @brief A special list with convenience methods 
     * for working with sets of particles.
     */
    struct CAPI_EXPORT ParticleList {
        int32_t *parts;
        int32_t nr_parts;
        int32_t size_parts;
        uint16_t flags;
        
        // frees the memory associated with the parts list.
        void free();

        /**
         * @brief Reserve enough storage for a given number of items.
         * 
         * @param _nr_parts number of items
         * @return HRESULT 
         */
        HRESULT reserve(size_t _nr_parts);
        
        // inserts the given id into the list, returns the index of the item. 
        uint16_t insert(int32_t item);

        /**
         * @brief Inserts the given particle into the list, returns the index of the item. 
         * 
         * @param particle particle to insert
         * @return uint16_t 
         */
        uint16_t insert(const ParticleHandle *particle);
        
        /**
         * @brief looks for the item with the given id and deletes it form the list
         * 
         * @param id id to remove
         * @return uint16_t 
         */
        uint16_t remove(int32_t id);
        
        /**
         * @brief inserts the contents of another list
         * 
         * @param other another list
         */
        void extend(const ParticleList &other);

        /**
         * @brief looks for the item at the given index and returns it if found, otherwise returns NULL
         * 
         * @param i index of lookup
         * @return ParticleHandle* 
         */
        ParticleHandle *item(const int32_t &i);

        // packs a variable number of particle ids into a new list
        static ParticleList *pack(size_t n, ...);

        /**
         * @brief returns an instance populated with all current particles
         * 
         * @return ParticleList* 
         */
        static ParticleList* all();

        FMatrix3 getVirial();
        FPTYPE getRadiusOfGyration();
        FVector3 getCenterOfMass();
        FVector3 getCentroid();
        FMatrix3 getMomentOfInertia();
        std::vector<FVector3> getPositions();
        std::vector<FVector3> getVelocities();
        std::vector<FVector3> getForces();

        /**
         * @brief Get the spherical coordinates of each particle
         * 
         * @param origin optional origin of coordinates; default is center of universe
         * @return std::vector<FVector3> 
         */
        std::vector<FVector3> sphericalPositions(FVector3 *origin=NULL);

        ParticleList();
        ParticleList(uint16_t init_size, uint16_t flags = PARTICLELIST_OWNDATA |PARTICLELIST_MUTABLE | PARTICLELIST_OWNSELF);
        ParticleList(ParticleHandle *part);
        ParticleList(std::vector<ParticleHandle*> particles);
        ParticleList(uint16_t nr_parts, int32_t *parts);
        ParticleList(const ParticleList &other);
        ~ParticleList();

        /**
         * @brief Get a JSON string representation
         * 
         * @return std::string 
         */
        std::string toString();

        /**
         * @brief Create from a JSON string representation
         * 
         * @param str 
         * @return ParticleList* 
         */
        static ParticleList *fromString(const std::string &str);
        
    };


};

#endif // _MDCORE_INCLUDE_TFPARTICLELIST_H_