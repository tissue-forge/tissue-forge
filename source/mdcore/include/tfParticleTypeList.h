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

#ifndef _MDCORE_INCLUDE_TFPARTICLETYPELIST_H_
#define _MDCORE_INCLUDE_TFPARTICLETYPELIST_H_

#include "tfParticleList.h"


namespace TissueForge { 


    struct ParticleType;
    struct ClusterParticleType;

    /**
     * @brief A special list with convenience methods 
     * for working with sets of particle types.
     */
    struct CAPI_EXPORT ParticleTypeList {
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
         * @brief Inserts the given particle type into the list, returns the index of the item. 
         * 
         * @param ptype 
         * @return uint16_t 
         */
        uint16_t insert(const ParticleType *ptype);
        
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
        void extend(const ParticleTypeList &other);

        /** Test whether the list has an id */
        bool has(const int32_t &pid);

        /** Test whether the list has a particle type */
        bool has(ParticleType *ptype);

        /** Test whether the list has a particle */
        bool has(ParticleHandle *part);

        /**
         * @brief looks for the item at the given index and returns it if found, otherwise returns NULL
         * 
         * @param i index of lookup
         * @return ParticleType* 
         */
        ParticleType *item(const int32_t &i);

        /** get an item at a given index */
        int32_t operator[](const size_t &i);

        /** returns the list as a vector */
        std::vector<int32_t> vector();

        /**
         * @brief returns a list populated with particles of all current particle types
         * 
         * @return ParticleList* 
         */
        ParticleList particles();

        /**
         * @brief returns an instance populated with all current particle types
         * 
         * @return ParticleTypeList* 
         */
        static ParticleTypeList all();

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

        ParticleTypeList();
        ParticleTypeList(uint16_t init_size, uint16_t flags = PARTICLELIST_OWNDATA |PARTICLELIST_MUTABLE | PARTICLELIST_OWNSELF);
        ParticleTypeList(ParticleType *ptype);
        ParticleTypeList(std::vector<ParticleType> ptypes);
        ParticleTypeList(std::vector<ParticleType*> ptypes);
        ParticleTypeList(uint16_t nr_parts, int32_t *ptypes);
        ParticleTypeList(const ParticleTypeList &other);
        ParticleTypeList(const std::vector<int32_t> &pids);
        ~ParticleTypeList();

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
         * @return ParticleTypeList* 
         */
        static ParticleTypeList *fromString(const std::string &str);
        
    };

};

#endif // _MDCORE_INCLUDE_TFPARTICLETYPELIST_H_