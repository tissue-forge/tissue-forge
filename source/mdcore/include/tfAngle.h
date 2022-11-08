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
 * @file tfAngle.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFANGLE_H_
#define _MDCORE_INCLUDE_TFANGLE_H_

#include <mdcore_config.h>
#include "tfPotential.h"


namespace TissueForge {


    namespace rendering {
        struct Style;
    }


    typedef enum AngleFlags {

        // none type angles are initial state, and can be
        // re-assigned if ref count is 1 (only owned by engine).
        ANGLE_NONE                   = 0,
        ANGLE_ACTIVE                 = 1 << 0
    } AngleFlags;

    struct AngleHandle;
    struct ParticleHandle;

    /**
     * @brief A bond concerning an angle
     * 
     * If you're building a model, you should probably instead be working with a 
     * AngleHandle. 
     */
    typedef struct CAPI_EXPORT Angle {

        uint32_t flags;

        /* ids of particles involved */
        int i, j, k;
        
        uint32_t id;

        uint64_t creation_time;

        /**
         * half life decay time for this bond.
         */
        FPTYPE half_life;

        /* potential energy required to break this bond */
        FPTYPE dissociation_energy;

        /* potential energy of this bond */
        FPTYPE potential_energy;

        /* id of the potential. */
        struct Potential *potential;

        struct rendering::Style *style;

        /**
         * @brief Get the default style
         * 
         * @return rendering::Style* 
         */
        static rendering::Style *styleDef();

        void init(Potential *potential, 
                  struct ParticleHandle *p1, 
                  struct ParticleHandle *p2, 
                  struct ParticleHandle *p3, 
                  uint32_t flags=0);

        /**
         * @brief Creates an angle bond. 
         * 
         * Automatically updates when running on a CUDA device. 
         * 
         * @param potential potential of the bond
         * @param p1 first outer particle
         * @param p2 center particle
         * @param p3 second outer particle
         * @param flags angle flags
         * @return AngleHandle* 
         */
        static AngleHandle *create(Potential *potential, 
                                   struct ParticleHandle *p1, 
                                   struct ParticleHandle *p2, 
                                   struct ParticleHandle *p3, 
                                   uint32_t flags=0);

        /**
         * @brief Get a JSON string representation
         * 
         * @return std::string 
         */
        std::string toString();

        /**
         * @brief Create from a JSON string representation. 
         * 
         * The returned angle is not automatically registered with the engine. 
         * 
         * @param str 
         * @return Angle* 
         */
        static Angle *fromString(const std::string &str);

    } Angle;

    /**
     * @brief A handle to an angle bond
     * 
     * This is a safe way to work with an angle bond. 
     */
    struct CAPI_EXPORT AngleHandle {
        int id;

        /**
         * @brief Gets the angle of this handle
         * 
         * @return Angle* 
         */
        Angle *get();

        /**
         * @brief Get a summary string of the angle
         * 
         * @return std::string 
         */
        std::string str();

        /**
         * @brief Check the validity of the handle
         * 
         * @return true if ok
         * @return false 
         */
        bool check();

        /**
         * @brief Destroy the angle. 
         * 
         * Automatically updates when running on a CUDA device. 
         */
        HRESULT destroy();

        /**
         * @brief Gets all angles in the universe
         * 
         * @return std::vector<AngleHandle*> 
         */
        static std::vector<AngleHandle*> items();

        /**
         * @brief Tests whether this bond decays
         * 
         * @return true when the bond should decay
         */
        bool decays();

        struct ParticleHandle *operator[](unsigned int index);

        FPTYPE getEnergy();
        std::vector<int32_t> getParts();
        Potential *getPotential();
        uint32_t getId();
        FPTYPE getDissociationEnergy();
        void setDissociationEnergy(const FPTYPE &dissociation_energy);
        FPTYPE getHalfLife();
        void setHalfLife(const FPTYPE &half_life);
        bool getActive();
        rendering::Style *getStyle();
        void setStyle(rendering::Style *style);
        FPTYPE getAge();

        AngleHandle() : id(-1) {}
        AngleHandle(const int &_id) : id(_id) {}
    };

    /**
     * @brief Destroys an angle
     * 
     * @param a angle to destroy
     */
    CAPI_FUNC(HRESULT) Angle_Destroy(Angle *a);

    /**
     * @brief Destroys all angles in the universe. 
     * 
     * Automatically updates when running on a CUDA device. 
     */
    CAPI_FUNC(HRESULT) Angle_DestroyAll();

    /* associated functions */
    
    /**
     * @brief Evaluate a list of angleed interactions
     *
     * @param a Pointer to an array of #angle.
     * @param N Nr of angles in @c b.
     * @param e Pointer to the #engine in which these angles are evaluated.
     * @param epot_out Pointer to a FPTYPE in which to aggregate the potential energy.
     */
    HRESULT angle_eval(struct Angle *a, int N, struct engine *e, FPTYPE *epot_out);

    /**
     * @brief Evaluate a list of angleed interactions
     *
     * @param a Pointer to an array of #angle.
     * @param N Nr of angles in @c b.
     * @param e Pointer to the #engine in which these angles are evaluated.
     * @param epot_out Pointer to a FPTYPE in which to aggregate the potential energy.
     *
     * This function differs from #angle_eval in that the forces are added to
     * the array @c f instead of directly in the particle data.
     */
    HRESULT angle_evalf(struct Angle *a, int N, struct engine *e, FPTYPE *f, FPTYPE *epot_out);

    /**
     * find all the angles that interact with the given particle id
     */
    std::vector<int32_t> Angle_IdsForParticle(int32_t pid);


};

#endif // _MDCORE_INCLUDE_TFANGLE_H_