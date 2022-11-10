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
 * @file tfDihedral.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFDIHEDRAL_H_
#define _MDCORE_INCLUDE_TFDIHEDRAL_H_

#include <mdcore_config.h>
#include <tfParticleList.h>


namespace TissueForge { 


    namespace rendering {
        struct Style;
    }


    typedef enum DihedralFlags {

        // none type dihedral are initial state, and can be
        // re-assigned if ref count is 1 (only owned by engine).
        DIHEDRAL_NONE                   = 0,
        DIHEDRAL_ACTIVE                 = 1 << 0
    } DihedralFlags;

    struct DihedralHandle;
    struct ParticleHandle;

    /** The dihedral structure */
    typedef struct CAPI_EXPORT Dihedral {

        uint32_t flags;

        /* ids of particles involved */
        int i, j, k, l;
        
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

        /* dihedral potential. */
        struct Potential *potential;

        struct rendering::Style *style;

        /**
         * @brief Get the default style
         * 
         * @return rendering::Style* 
         */
        static rendering::Style *styleDef();

        void init(Potential *potential, 
                  ParticleHandle *p1, 
                  ParticleHandle *p2, 
                  ParticleHandle *p3, 
                  ParticleHandle *p4);

        /**
         * @brief Creates a dihedral bond
         * 
         * @param potential potential of the bond
         * @param p1 first outer particle
         * @param p2 first center particle
         * @param p3 second center particle
         * @param p4 second outer particle
         * @return DihedralHandle* 
         */
        static DihedralHandle *create(Potential *potential, 
                                      ParticleHandle *p1, 
                                      ParticleHandle *p2, 
                                      ParticleHandle *p3, 
                                      ParticleHandle *p4);

        /**
         * @brief Get a JSON string representation
         * 
         * @return std::string 
         */
        std::string toString();

        /**
         * @brief Create from a JSON string representation. 
         * 
         * The returned dihedral is not automatically registered with the engine. 
         * 
         * @param str 
         * @return Dihedral* 
         */
        static Dihedral *fromString(const std::string &str);

    } Dihedral;

    /**
     * @brief A handle to a dihedral bond
     * 
     * This is a safe way to work with a dihedral bond. 
     */
    struct CAPI_EXPORT DihedralHandle {
        int id;

        /**
         * @brief Gets the dihedral of this handle
         * 
         * @return Dihedral* 
         */
        Dihedral *get();

        /**
         * @brief Get a summary string of the dihedral
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
         * @brief Destroy the dihedral
         */
        HRESULT destroy();

        /**
         * @brief Gets all dihedrals in the universe
         */
        static std::vector<DihedralHandle> items();

        /**
         * @brief Tests whether this bond decays
         * 
         * @return true when the bond should decay
         */
        bool decays();

        ParticleHandle *operator[](unsigned int index);

        /** Test whether the bond has an id */
        bool has(const int32_t &pid);

        /** Test whether the bond has a particle */
        bool has(ParticleHandle *part);

        FPTYPE getEnergy();
        std::vector<int32_t> getParts();
        ParticleList getPartList();
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

        DihedralHandle() : id(-1) {}
        DihedralHandle(const int &_id) : id(_id) {}
    };

    /**
     * @brief Destroys a dihedral
     * 
     * @param d dihedral to destroy
     */
    CAPI_FUNC(HRESULT) Dihedral_Destroy(Dihedral *d);

    /**
     * @brief Destroys all dihedrals in the universe
     */
    CAPI_FUNC(HRESULT) Dihedral_DestroyAll();

    /* associated functions */

    /**
     * @brief Evaluate a list of dihedraled interactions
     *
     * @param b Pointer to an array of #dihedral.
     * @param N Nr of dihedrals in @c b.
     * @param e Pointer to the #engine in which these dihedrals are evaluated.
     * @param epot_out Pointer to a FPTYPE in which to aggregate the potential energy.
     */
    HRESULT dihedral_eval(struct Dihedral *d, int N, struct engine *e, FPTYPE *epot_out);

    /**
     * @brief Evaluate a list of dihedraled interactions
     *
     * @param b Pointer to an array of #dihedral.
     * @param N Nr of dihedrals in @c b.
     * @param e Pointer to the #engine in which these dihedrals are evaluated.
     * @param epot_out Pointer to a FPTYPE in which to aggregate the potential energy.
     *
     * This function differs from #dihedral_eval in that the forces are added to
     * the array @c f instead of directly in the particle data.
     */
    HRESULT dihedral_evalf(struct Dihedral *d, int N, struct engine *e, FPTYPE *f, FPTYPE *epot_out);

    /**
     * find all the dihedrals that interact with the given particle id
     */
    std::vector<int32_t> Dihedral_IdsForParticle(int32_t pid);

};


inline bool operator< (const TissueForge::DihedralHandle& lhs, const TissueForge::DihedralHandle& rhs) { return lhs.id < rhs.id; }
inline bool operator> (const TissueForge::DihedralHandle& lhs, const TissueForge::DihedralHandle& rhs) { return rhs < lhs; }
inline bool operator<=(const TissueForge::DihedralHandle& lhs, const TissueForge::DihedralHandle& rhs) { return !(lhs > rhs); }
inline bool operator>=(const TissueForge::DihedralHandle& lhs, const TissueForge::DihedralHandle& rhs) { return !(lhs < rhs); }
inline bool operator==(const TissueForge::DihedralHandle& lhs, const TissueForge::DihedralHandle& rhs) { return lhs.id == rhs.id; }
inline bool operator!=(const TissueForge::DihedralHandle& lhs, const TissueForge::DihedralHandle& rhs) { return !(lhs == rhs); }

inline std::ostream &operator<<(std::ostream& os, const TissueForge::DihedralHandle &h)
{
    os << h.str().c_str();
    return os;
}

#endif // _MDCORE_INCLUDE_TFDIHEDRAL_H_