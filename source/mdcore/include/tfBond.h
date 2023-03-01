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
 * @file tfBond.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFBOND_H_
#define _MDCORE_INCLUDE_TFBOND_H_

#include <mdcore_config.h>
#include <tfParticleList.h>


namespace TissueForge {


    namespace rendering {
        struct Style;
    }


    typedef enum BondFlags {

        // none type bonds are initial state, and can be
        // re-assigned if ref count is 1 (only owned by engine).
        BOND_NONE                   = 0,
        // a non-active and will be over-written in the
        // next bond_alloc call.
        BOND_ACTIVE                 = 1 << 0,
    } BondFlags;

    // list of pairs...
    struct Pair {
        int32_t i;
        int32_t j;
    };

    typedef std::vector<Pair> PairList;
    struct BondHandle;
    struct ParticleHandle;

    /**
     * @brief Bonds apply a potential to a particular set of particles. 
     * 
     * If you're building a model, you should probably instead be working with a 
     * BondHandle. 
     */
    typedef struct CAPI_EXPORT Bond {

        uint32_t flags;

        /* id of particles involved */
        int32_t i, j;
        
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

        struct Potential *potential;
        
        struct rendering::Style *style;

        /**
         * @brief Get the default style
         */
        static rendering::Style *styleDef();

        /**
         * @brief Construct a new bond handle and underlying bond. 
         * 
         * Automatically updates when running on a CUDA device. 
         * 
         * @param potential bond potential
         * @param i ith particle
         * @param j jth particle
         * @param half_life bond half life
         * @param dissociation_energy dissociation energy
         * @param flags bond flags
         */
        static BondHandle *create(
            struct Potential *potential, 
            ParticleHandle *i, 
            ParticleHandle *j, 
            FPTYPE *half_life=NULL, 
            FPTYPE *dissociation_energy=NULL, 
            uint32_t flags=0
        );

        /**
         * @brief Get a JSON string representation
         */
        std::string toString();

        /**
         * @brief Create from a JSON string representation. 
         * 
         * The returned bond is not automatically registered with the engine. 
         * 
         * @param str 
         */
        static Bond *fromString(const std::string &str);

    } Bond;

    struct ParticleType;
    struct ParticleList;

    /**
     * @brief Handle to a bond
     * 
     * This is a safe way to work with a bond. 
     */
    struct CAPI_EXPORT BondHandle {
        int32_t id;
        
        /**
         * @brief Gets the underlying bond
         * 
         * @return bond, if available
         */
        TissueForge::Bond *get();

        /**
         * @brief Construct a new bond handle and do nothing
         * Subsequent usage will require a call to 'init'
         * 
         */
        BondHandle() : id(-1) {};

        /**
         * @brief Construct a new bond handle from an existing bond id
         * 
         * @param id id of existing bond
         */
        BondHandle(int id);

        /**
         * @brief Construct a new bond handle and underlying bond. 
         * 
         * Automatically updates when running on a CUDA device. 
         * 
         * @param potential bond potential
         * @param i id of ith particle
         * @param j id of jth particle
         * @param half_life bond half life
         * @param bond_energy bond energy
         * @param flags bond flags
         */
        BondHandle(
            struct TissueForge::Potential *potential, 
            int32_t i, 
            int32_t j, 
            FPTYPE half_life, 
            FPTYPE bond_energy, 
            uint32_t flags
        );

        /**
         * @brief For initializing a bond after constructing with default constructor. 
         * 
         * Automatically updates when running on a CUDA device. 
         * 
         * @param pot bond potential
         * @param p1 ith particle
         * @param p2 jth particle
         * @param half_life bond half life
         * @param bond_energy bond energy
         * @param flags bond flags
         */
        HRESULT init(
            TissueForge::Potential *pot, 
            TissueForge::ParticleHandle *p1, 
            TissueForge::ParticleHandle *p2, 
            const FPTYPE &half_life=-FPTYPE_ONE, 
            const FPTYPE &bond_energy=-FPTYPE_ONE, 
            uint32_t flags=0
        );
        
        /**
         * @brief Get a summary string of the bond
         */
        std::string str() const;

        /**
         * @brief Check the validity of the handle
         * 
         * @return true if ok
         * @return false 
         */
        bool check();
        
        /**
         * @brief Apply bonds to a list of particles. 
         * 
         * Automatically updates when running on a CUDA device. 
         * 
         * @param pot the potential of the created bonds
         * @param parts list of particles
         * @param cutoff cutoff distance of particles that are bonded
         * @param ppairs type pairs of bonds
         * @param half_life bond half life
         * @param bond_energy bond energy
         * @param flags bond flags
         * @return created bonds
         */
        static std::vector<BondHandle> pairwise(
            TissueForge::Potential* pot,
            TissueForge::ParticleList &parts,
            const FPTYPE &cutoff,
            std::vector<std::pair<TissueForge::ParticleType*, TissueForge::ParticleType*>* > *ppairs,
            const FPTYPE &half_life,
            const FPTYPE &bond_energy,
            uint32_t flags
        );
        
        /**
         * @brief Destroy the bond. 
         * 
         * Automatically updates when running on a CUDA device. 
         */
        HRESULT destroy();

        /**
         * @brief Gets all bonds in the universe
         */
        static std::vector<BondHandle> items();

        /**
         * @brief Tests whether this bond decays
         * 
         * @return true when the bond should decay
         */
        bool decays();

        TissueForge::ParticleHandle *operator[](unsigned int index);

        /** Test whether the bond has an id */
        bool has(const int32_t &pid);

        /** Test whether the bond has a particle */
        bool has(ParticleHandle *part);

        /** Get the current length */
        FloatP_t getLength();
        
        /** Get the energy */
        FPTYPE getEnergy();

        /** Get the particle ids */
        std::vector<int32_t> getParts();

        /** Get the particle list */
        ParticleList getPartList();

        /** Get the potential */
        TissueForge::Potential *getPotential();

        /** Get the id */
        uint32_t getId();

        /** Get the dissociation energy */
        FPTYPE getDissociationEnergy();

        /** Set the dissociation energy */
        void setDissociationEnergy(const FPTYPE &dissociation_energy);

        /** Get the half life */
        FPTYPE getHalfLife();

        /** Set the half life */
        void setHalfLife(const FPTYPE &half_life);

        /** Get the style */
        rendering::Style *getStyle();

        /** Set the style */
        void setStyle(rendering::Style *style);

        /** Get the age */
        FPTYPE getAge();

    private:

        HRESULT _init(
            uint32_t flags, 
            int32_t i, 
            int32_t j, 
            FPTYPE half_life, 
            FPTYPE bond_energy, 
            struct Potential *potential
        );
    };

    bool contains_bond(const std::vector<BondHandle> &bonds, int a, int b);

    /**
     * deletes, marks a bond ready for deleteion, removes the potential,
     * other vars, clears the bond, and makes is ready to be
     * over-written. 
     * 
     * Automatically updates when running on a CUDA device. 
     */
    CAPI_FUNC(HRESULT) Bond_Destroy(Bond *b);

    /**
     * @brief Deletes all bonds in the universe. 
     * 
     * Automatically updates when running on a CUDA device. 
     */
    CAPI_FUNC(HRESULT) Bond_DestroyAll();

    HRESULT Bond_Energy (Bond *b, FPTYPE *epot_out);

    /* associated functions */

    /**
     * @brief Evaluate a list of bonded interactions
     *
     * @param b Pointer to an array of #bond.
     * @param N Nr of bonds in @c b.
     * @param e Pointer to the #engine in which these bonds are evaluated.
     * @param epot_out Pointer to a FPTYPE in which to aggregate the potential energy.
     */
    CAPI_FUNC(HRESULT) bond_eval (Bond *b, int N, struct engine *e, FPTYPE *epot_out);

    /**
     * @brief Evaluate a list of bonded interactions
     *
     * @param bonds Pointer to an array of #bond.
     * @param N Nr of bonds in @c b.
     * @param e Pointer to the #engine in which these bonds are evaluated.
     * @param forces An array of @c FPTYPE in which to aggregate the resulting forces.
     * @param epot_out Pointer to a FPTYPE in which to aggregate the potential energy.
     * 
     * This function differs from #bond_eval in that the forces are added to
     * the array @c f instead of directly in the particle data.
     */
    CAPI_FUNC(HRESULT) bond_evalf (Bond *b, int N, struct engine *e, FPTYPE *f, FPTYPE *epot_out);


    /**
     * find all the bonds that interact with the given particle id
     */
    std::vector<int32_t> Bond_IdsForParticle(int32_t pid);

    int insert_bond(
        std::vector<BondHandle> &bonds, 
        int a, 
        int b,
        Potential *pot, 
        ParticleList *parts
    );

    inline bool operator< (const TissueForge::BondHandle& lhs, const TissueForge::BondHandle& rhs) { return lhs.id < rhs.id; }
    inline bool operator> (const TissueForge::BondHandle& lhs, const TissueForge::BondHandle& rhs) { return rhs < lhs; }
    inline bool operator<=(const TissueForge::BondHandle& lhs, const TissueForge::BondHandle& rhs) { return !(lhs > rhs); }
    inline bool operator>=(const TissueForge::BondHandle& lhs, const TissueForge::BondHandle& rhs) { return !(lhs < rhs); }
    inline bool operator==(const TissueForge::BondHandle& lhs, const TissueForge::BondHandle& rhs) { return lhs.id == rhs.id; }
    inline bool operator!=(const TissueForge::BondHandle& lhs, const TissueForge::BondHandle& rhs) { return !(lhs == rhs); }


};


inline std::ostream &operator<<(std::ostream& os, const TissueForge::BondHandle &h)
{
    os << h.str().c_str();
    return os;
}

#endif // _MDCORE_INCLUDE_TFBOND_H_