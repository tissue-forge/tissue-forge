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
 * @file tfPotential.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TFPOTENTIAL_H_
#define _MDCORE_INCLUDE_TFPOTENTIAL_H_

#include "tf_platform.h"
#include "tf_fptype.h"
#include <io/tf_io.h>

#include <limits>
#include <utility>
#include <vector>

/* some constants */
#define potential_degree                    5
#define potential_chunk                     (potential_degree+3)
#define potential_ivalsa                    1
#define potential_ivalsb                    10
#define potential_N                         100
#define potential_align                     64
#define potential_ivalsmax                  3048

#define potential_escale                    (0.079577471545947667882)
// #define potential_escale                    1.0


namespace TissueForge {


    /* potential flags */

    enum PotentialFlags {
        POTENTIAL_NONE            = 0,
        POTENTIAL_LJ126           = 1 << 0,
        POTENTIAL_EWALD           = 1 << 1,
        POTENTIAL_COULOMB         = 1 << 2,
        POTENTIAL_SINGLE          = 1 << 3,

        /** flag defined for r^2 input */
        POTENTIAL_R2              = 1 << 4,

        /** potential defined for r input (no sqrt) */
        POTENTIAL_R               = 1 << 5,

        /** potential defined for angle */
        POTENTIAL_ANGLE           = 1 << 6,

        /** potential defined for harmonic */
        POTENTIAL_HARMONIC        = 1 << 7,

        POTENTIAL_DIHEDRAL        = 1 << 8,

        /** potential defined for switch */
        POTENTIAL_SWITCH          = 1 << 9,

        POTENTIAL_REACTIVE        = 1 << 10,

        /**
         * Scaled functions take a (r0/r)^2 argument instead of an r^2,
         * they include the rest length r0, such that r0/r yields a
         * force = 0.
         */
        POTENTIAL_SCALED          = 1 << 11,

        /**
         * potential shifted by x value,
         */
        POTENTIAL_SHIFTED         = 1 << 12,

        /**
         * potential is valid for bound particles, if un-set,
         * potential is for free particles.
         */
        POTENTIAL_BOUND           = 1 << 13,

        // sum of constituent potentials
        POTENTIAL_SUM             = 1 << 14, 

        /** unbound potential with long-range periodicity */
        POTENTIAL_PERIODIC        = 1 << 15, 

        /* Coulomb reciprocated */
        POTENTIAL_COULOMBR        = 1 << 16
    };

    enum PotentialKind {
        // standard interpolated potential kind
        POTENTIAL_KIND_POTENTIAL,
        
        // dissipative particle dynamics kind
        POTENTIAL_KIND_DPD, 

        // explicit potential by particles
        POTENTIAL_KIND_BYPARTICLES,

        // combination of two constituent potentials
        POTENTIAL_KIND_COMBINATION
    };


    /**
     * @brief Potential function on a particle. 
     * 
     * Includes pre-computed relative position and distance of an arbitrary point w.r.t. ith particle. 
     * 
     * Computes the potential and force in global frame on the ith particle. 
     * 
     */
    typedef void (*PotentialEval_ByParticle) (
        struct Potential *p, 
        struct Particle *part_i, 
        FPTYPE *dx, 
        FPTYPE r2, 
        FPTYPE *e, 
        FPTYPE *f
    );

    /**
     * @brief Pair potential function. 
     * 
     * Includes pre-computed relative position and distance of jth particle w.r.t. ith particle. 
     * 
     * Computes the potential and force in global frame on the ith particle. 
     * 
     */
    typedef void (*PotentialEval_ByParticles) (
        struct Potential *p, 
        struct Particle *part_i, 
        struct Particle *part_j, 
        FPTYPE *dx, 
        FPTYPE r2, 
        FPTYPE *e, 
        FPTYPE *f
    );

    /**
     * @brief Like PotentialEval_ByParticles, but with three particles
     * 
     */
    typedef void (*PotentialEval_ByParticles3)(
        struct Potential *p, 
        struct Particle *part_i, 
        struct Particle *part_j, 
        struct Particle *part_k, 
        FPTYPE ctheta, 
        FPTYPE *e, 
        FPTYPE *fi, 
        FPTYPE *fk
    );

    /**
     * @brief Like PotentialEval_ByParticles, but with four particles
     * 
     */
    typedef void (*PotentialEval_ByParticles4)(
        struct Potential *p, 
        struct Particle *part_i, 
        struct Particle *part_j, 
        struct Particle *part_k, 
        struct Particle *part_l, 
        FPTYPE cphi, 
        FPTYPE *e, 
        FPTYPE *fi, 
        FPTYPE *fl
    );

    typedef struct Potential* (*PotentialCreate) (
        struct Potential *partial_potential,
        struct ParticleType *a, 
        struct ParticleType *b
    );

    /**
     * @brief Callback issues when potential is cleared. 
     * 
     */
    typedef void (*PotentialClear) (struct Potential* p);


    /**
     * @brief A Potential object is a compiled interpolation of a given function. The 
     * Universe applies potentials to particles to calculate the net force on them. 
     * 
     * For performance reasons, Tissue Forge implements potentials as 
     * interpolations, which can be much faster than evaluating the function directly. 
     * 
     * A potential can be treated just like any callable object. 
     */
    struct CAPI_EXPORT Potential {
        uint32_t kind;

        /** Flags. */
        uint32_t flags;


        /** Coefficients for the interval transform. */
        FPTYPE alpha[4];

        /** The coefficients. */
        FPTYPE *c;

        FPTYPE r0_plusone;

        /** Interval edges. */
        FPTYPE a, b;

        /** potential scaling constant */
        FPTYPE mu;

        /** coordinate offset */
        FPTYPE offset[3];

        /** Nr of intervals. */
        int n;

        PotentialCreate create_func;
        PotentialClear clear_func;

        PotentialEval_ByParticle eval_bypart;
        PotentialEval_ByParticles eval_byparts;
        PotentialEval_ByParticles3 eval_byparts3;
        PotentialEval_ByParticles4 eval_byparts4;

        Potential *pca, *pcb;

        /**
         * pointer to what kind of potential this is.
         */
        const char* name;

        Potential();

        FPTYPE operator()(const FPTYPE &r, const FPTYPE &r0=-1.0);
        FPTYPE operator()(const std::vector<FPTYPE>& r);
        FPTYPE operator()(struct ParticleHandle* pi, const FVector3 &pt);
        FPTYPE operator()(struct ParticleHandle* pi, struct ParticleHandle* pj);
        FPTYPE operator()(struct ParticleHandle* pi, struct ParticleHandle* pj, struct ParticleHandle* pk);
        FPTYPE operator()(struct ParticleHandle* pi, struct ParticleHandle* pj, struct ParticleHandle* pk, struct ParticleHandle* pl);
        FPTYPE force(FPTYPE r, FPTYPE ri=-1.0, FPTYPE rj=-1.0);
        std::vector<FPTYPE> force(const std::vector<FPTYPE>& r);
        std::vector<FPTYPE> force(struct ParticleHandle* pi, const FVector3 &pt);
        std::vector<FPTYPE> force(struct ParticleHandle* pi, struct ParticleHandle* pj);
        std::pair<std::vector<FPTYPE>, std::vector<FPTYPE> > force(struct ParticleHandle* pi, struct ParticleHandle* pj, struct ParticleHandle* pk);
        std::pair<std::vector<FPTYPE>, std::vector<FPTYPE> > force(struct ParticleHandle* pi, struct ParticleHandle* pj, struct ParticleHandle* pk, struct ParticleHandle* pl);

        std::vector<Potential*> constituents();

        Potential& operator+(const Potential& rhs);

        /**
         * @brief Get a JSON string representation
         * 
         * @return std::string 
         */
        virtual std::string toString();

        /**
         * @brief Create from a JSON string representation
         * 
         * @param str 
         * @return Potential* 
         */
        static Potential *fromString(const std::string &str);

        /**
         * @brief Creates a 12-6 Lennard-Jones potential. 
         * 
         * The Lennard Jones potential has the form:
         * 
         * @f[
         * 
         *      \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) 
         * 
         * @f]
         * 
         * @param min The smallest radius for which the potential will be constructed.
         * @param max The largest radius for which the potential will be constructed.
         * @param A The first parameter of the Lennard-Jones potential.
         * @param B The second parameter of the Lennard-Jones potential.
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
         * @return Potential* 
         */
        static Potential *lennard_jones_12_6(FPTYPE min, FPTYPE max, FPTYPE A, FPTYPE B, FPTYPE *tol=NULL);

        /**
         * @brief Creates a potential of the sum of a 12-6 Lennard-Jones potential and a shifted Coulomb potential. 
         * 
         * The 12-6 Lennard Jones - Coulomb potential has the form:
         * 
         * @f[
         * 
         *      \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) + q \left( \frac{1}{r} - \frac{1}{max} \right)
         * 
         * @f]
         * 
         * @param min The smallest radius for which the potential will be constructed.
         * @param max The largest radius for which the potential will be constructed.
         * @param A The first parameter of the Lennard-Jones potential.
         * @param B The second parameter of the Lennard-Jones potential.
         * @param q The charge scaling of the potential.
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
         * @return Potential* 
         */
        static Potential *lennard_jones_12_6_coulomb(FPTYPE min, FPTYPE max, FPTYPE A, FPTYPE B, FPTYPE q, FPTYPE *tol=NULL);

        /**
         * @brief Creates a real-space Ewald potential. 
         * 
         * The Ewald potential has the form:
         * 
         * @f[
         * 
         *      q \frac{\mathrm{erfc}\, ( \kappa r)}{r}
         * 
         * @f]
         * 
         * @param min The smallest radius for which the potential will be constructed.
         * @param max The largest radius for which the potential will be constructed.
         * @param q The charge scaling of the potential.
         * @param kappa The screening distance of the Ewald potential.
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
         * @param periodicOrder Order of lattice periodicity along all periodic dimensions. Defaults to 0. 
         * @return Potential* 
         */
        static Potential *ewald(FPTYPE min, FPTYPE max, FPTYPE q, FPTYPE kappa, FPTYPE *tol=NULL, unsigned int *periodicOrder=NULL);

        /**
         * @brief Creates a Coulomb potential. 
         * 
         * The Coulomb potential has the form:
         * 
         * @f[
         * 
         *      \frac{q}{r}
         * 
         * @f]
         * 
         * @param q The charge scaling of the potential. 
         * @param min The smallest radius for which the potential will be constructed. Default is 0.01. 
         * @param max The largest radius for which the potential will be constructed. Default is 2.0. 
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001 * (max - min). 
         * @param periodicOrder Order of lattice periodicity along all periodic dimensions. Defaults to 0. 
         * @return Potential* 
         */
        static Potential *coulomb(FPTYPE q, FPTYPE *min=NULL, FPTYPE *max=NULL, FPTYPE *tol=NULL, unsigned int *periodicOrder=NULL);

        /**
         * @brief Creates a Coulomb reciprocal potential. 
         * 
         * The Coulomb reciprocal potential has the form: 
         * 
         * @f[
         * 
         *      \frac{\pi q}{V} \sum_{||\mathbf{m}|| \neq 0} \frac{1}{||\mathbf{m}||^2} \exp \left( \left( i \mathbf{r}_{jk} - \left( \frac{\pi}{\kappa} \right)^{2} \mathbf{m} \right) \cdot \mathbf{m} \right)
         * 
         * @f]
         * 
         * Here @f$ V @f$ is the volume of the domain and @f$ \mathbf{m} @f$ is a reciprocal vector of the domain. 
         * 
         * @param q Charge scaling of the potential. 
         * @param kappa Screening distance.
         * @param min Smallest radius for which the potential will be constructed. 
         * @param max Largest radius for which the potential will be constructed. 
         * @param modes Number of Fourier modes along each periodic dimension. Default is 1. 
         * @return Potential* 
         */
        static Potential* coulombR(FPTYPE q, FPTYPE kappa, FPTYPE min, FPTYPE max, unsigned int* modes=NULL);

        /**
         * @brief Creates a harmonic bond potential. 
         * 
         * The harmonic potential has the form: 
         * 
         * @f[
         * 
         *      k \left( r-r_0 \right)^2
         * 
         * @f]
         * 
         * @param k The energy of the bond.
         * @param r0 The bond rest length.
         * @param min The smallest radius for which the potential will be constructed. Defaults to @f$ r_0 - r_0 / 2 @f$.
         * @param max The largest radius for which the potential will be constructed. Defaults to @f$ r_0 + r_0 /2 @f$.
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to @f$ 0.01 \abs(max-min) @f$.
         * @return Potential* 
         */
        static Potential *harmonic(FPTYPE k, FPTYPE r0, FPTYPE *min=NULL, FPTYPE *max=NULL, FPTYPE *tol=NULL);

        /**
         * @brief Creates a linear potential. 
         * 
         * The linear potential has the form:
         * 
         * @f[
         * 
         *      k r
         * 
         * @f]
         * 
         * @param k interaction strength; represents the potential energy peak value.
         * @param min The smallest radius for which the potential will be constructed. Defaults to 0.0.
         * @param max The largest radius for which the potential will be constructed. Defaults to 10.0.
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001.
         * @return Potential* 
         */
        static Potential *linear(FPTYPE k, FPTYPE *min=NULL, FPTYPE *max=NULL, FPTYPE *tol=NULL);

        /**
         * @brief Creates a harmonic angle potential. 
         * 
         * The harmonic angle potential has the form: 
         * 
         * @f[
         * 
         *      k \left(\theta-\theta_{0} \right)^2
         * 
         * @f]
         * 
         * @param k The energy of the angle.
         * @param theta0 The minimum energy angle.
         * @param min The smallest angle for which the potential will be constructed. Defaults to zero. 
         * @param max The largest angle for which the potential will be constructed. Defaults to PI. 
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.005 * (max - min). 
         * @return Potential* 
         */
        static Potential *harmonic_angle(FPTYPE k, FPTYPE theta0, FPTYPE *min=NULL, FPTYPE *max=NULL, FPTYPE *tol=NULL);

        /**
         * @brief Creates a harmonic dihedral potential. 
         * 
         * The harmonic dihedral potential has the form:
         * 
         * @f[
         * 
         *      k \left( \theta - \delta \right) ^2
         * 
         * @f]
         * 
         * @param k energy of the dihedral.
         * @param delta minimum energy dihedral. 
         * @param min The smallest angle for which the potential will be constructed. Defaults to zero. 
         * @param max The largest angle for which the potential will be constructed. Defaults to PI. 
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.005 * (max - min). 
         * @return Potential* 
         */
        static Potential *harmonic_dihedral(FPTYPE k, FPTYPE delta, FPTYPE *min=NULL, FPTYPE *max=NULL, FPTYPE *tol=NULL);

        /**
         * @brief Creates a cosine dihedral potential. 
         * 
         * The cosine dihedral potential has the form:
         * 
         * @f[
         * 
         *      k \left( 1 + \cos( n \theta-\delta ) \right)
         * 
         * @f]
         * 
         * @param k energy of the dihedral.
         * @param n multiplicity of the dihedral.
         * @param delta minimum energy dihedral. 
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.01. 
         * @return Potential* 
         */
        static Potential *cosine_dihedral(FPTYPE k, int n, FPTYPE delta, FPTYPE *tol=NULL);

        /**
         * @brief Creates a well potential. 
         * 
         * Useful for binding a particle to a region.
         * 
         * The well potential has the form: 
         * 
         * @f[
         * 
         *      \frac{k}{\left(r_0 - r\right)^{n}}
         * 
         * @f]
         * 
         * @param k potential prefactor constant, should be decreased for larger n.
         * @param n exponent of the potential, larger n makes a sharper potential.
         * @param r0 The extents of the potential, length units. Represents the maximum extents that a two objects connected with this potential should come apart.
         * @param min The smallest radius for which the potential will be constructed. Defaults to zero.
         * @param max The largest radius for which the potential will be constructed. Defaults to r0.
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.01 * abs(min-max).
         * @return Potential* 
         */
        static Potential *well(FPTYPE k, FPTYPE n, FPTYPE r0, FPTYPE *min=NULL, FPTYPE *max=NULL, FPTYPE *tol=NULL);

        /**
         * @brief Creates a generalized Lennard-Jones potential.
         * 
         * The generalized Lennard-Jones potential has the form:
         * 
         * @f[
         * 
         *      \frac{\epsilon}{n-m} \left[ m \left( \frac{r_0}{r} \right)^n - n \left( \frac{r_0}{r} \right)^m \right]
         * 
         * @f]
         * 
         * @param e effective energy of the potential. 
         * @param m order of potential. Defaults to 3
         * @param n order of potential. Defaults to 2*m.
         * @param k mimumum of the potential. Defaults to 1.
         * @param r0 mimumum of the potential. Defaults to 1. 
         * @param min The smallest radius for which the potential will be constructed. Defaults to 0.05 * r0.
         * @param max The largest radius for which the potential will be constructed. Defaults to 5 * r0.
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.01.
         * @param shifted Flag for whether using a shifted potential. Defaults to true. 
         * @return Potential* 
         */
        static Potential *glj(FPTYPE e, FPTYPE *m=NULL, FPTYPE *n=NULL, FPTYPE *k=NULL, FPTYPE *r0=NULL, FPTYPE *min=NULL, FPTYPE *max=NULL, FPTYPE *tol=NULL, bool *shifted=NULL);

        /**
         * @brief Creates a Morse potential. 
         * 
         * The Morse potential has the form:
         * 
         * @f[
         * 
         *      d \left(1 - e^{ -a \left(r - r_0 \right) } \right)^2
         * 
         * @f]
         * 
         * @param d well depth. Defaults to 1.0.
         * @param a potential width. Defaults to 6.0.
         * @param r0 equilibrium distance. Defaults to 0.0. 
         * @param min The smallest radius for which the potential will be constructed. Defaults to 0.0001.
         * @param max The largest radius for which the potential will be constructed. Defaults to 3.0.
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001.
         * @param shifted Flag for whether using a shifted potential. Defaults to true.
         * @return Potential* 
         */
        static Potential *morse(FPTYPE *d=NULL, FPTYPE *a=NULL, FPTYPE *r0=NULL, FPTYPE *min=NULL, FPTYPE *max=NULL, FPTYPE *tol=NULL, bool *shifted=NULL);

        /**
         * @brief Creates an overlapping-sphere potential from :cite:`Osborne:2017hk`. 
         * 
         * The overlapping-sphere potential has the form: 
         * 
         * @f[
         *      \mu_{ij} s_{ij}(t) \hat{\mathbf{r}}_{ij} \log \left( 1 + \frac{||\mathbf{r}_{ij}|| - s_{ij}(t)}{s_{ij}(t)} \right) 
         *          \text{ if } ||\mathbf{r}_{ij}|| < s_{ij}(t) ,
         * @f]
         * 
         * @f[
         *      \mu_{ij}\left(||\mathbf{r}_{ij}|| - s_{ij}(t)\right) \hat{\mathbf{r}}_{ij} \exp \left( -k_c \frac{||\mathbf{r}_{ij}|| - s_{ij}(t)}{s_{ij}(t)} \right) 
         *          \text{ if } s_{ij}(t) \leq ||\mathbf{r}_{ij}|| \leq r_{max} ,
         * @f]
         * 
         * @f[
         *      0 \text{ otherwise} .
         * @f]
         * 
         * Osborne refers to @f$ \mu_{ij} @f$ as a "spring constant", this 
         * controls the size of the force, and is the potential energy peak value. 
         * @f$ \hat{\mathbf{r}}_{ij} @f$ is the unit vector from particle 
         * @f$ i @f$ center to particle @f$ j @f$ center, @f$ k_C @f$ is a 
         * parameter that defines decay of the attractive force. Larger values of 
         * @f$ k_C @f$ result in a shaper peaked attraction, and thus a shorter 
         * ranged force. @f$ s_{ij}(t) @f$ is the is the sum of the radii of the 
         * two particles.
         * 
         * @param mu interaction strength, represents the potential energy peak value. Defaults to 1.0.
         * @param kc decay strength of long range attraction. Larger values make a shorter ranged function. Defaults to 1.0.
         * @param kh Optionally add a harmonic long-range attraction, same as :meth:`glj` function. Defaults to 0.0.
         * @param r0 Optional harmonic rest length, only used if `kh` is non-zero. Defaults to 0.0.
         * @param min The smallest radius for which the potential will be constructed. Defaults to 0.001.
         * @param max The largest radius for which the potential will be constructed. Defaults to 10.0.
         * @param tol The tolerance to which the interpolation should match the exact potential. Defaults to 0.001.
         * @return Potential* 
         */
        static Potential *overlapping_sphere(FPTYPE *mu=NULL, FPTYPE *kc=NULL, FPTYPE *kh=NULL, FPTYPE *r0=NULL, FPTYPE *min=NULL, FPTYPE *max=NULL, FPTYPE *tol=NULL);
        
        /**
         * @brief Creates a power potential. 
         * 
         * The power potential the general form of many of the potential 
         * functions, such as :meth:`linear`, etc. power has the form:
         * 
         * @f[
         * 
         *      k \lvert r-r_0 \rvert ^{\alpha}
         * 
         * @f]
         * 
         * @param k interaction strength, represents the potential energy peak value. Defaults to 1
         * @param r0 potential rest length, zero of the potential, defaults to 0.
         * @param alpha Exponent, defaults to 1.
         * @param min minimal value potential is computed for, defaults to r0 / 2.
         * @param max cutoff distance, defaults to 3 * r0.
         * @param tol Tolerance, defaults to 0.01.
         * @return Potential* 
         */
        static Potential *power(FPTYPE *k=NULL, FPTYPE *r0=NULL, FPTYPE *alpha=NULL, FPTYPE *min=NULL, FPTYPE *max=NULL, FPTYPE *tol=NULL);

        /**
         * @brief Creates a Dissipative Particle Dynamics potential. 
         * 
         * The Dissipative Particle Dynamics force has the form: 
         * 
         * @f[
         * 
         *      \mathbf{F}_{ij} = \mathbf{F}^C_{ij} + \mathbf{F}^D_{ij} + \mathbf{F}^R_{ij}
         * 
         * @f]
         * 
         * The conservative force is: 
         * 
         * @f[
         * 
         *      \mathbf{F}^C_{ij} = \alpha \left(1 - \frac{r_{ij}}{r_c}\right) \mathbf{e}_{ij}
         * 
         * @f]
         * 
         * The dissapative force is:
         * 
         * @f[
         * 
         *      \mathbf{F}^D_{ij} = -\gamma \left(1 - \frac{r_{ij}}{r_c}\right)^{2}(\mathbf{e}_{ij} \cdot \mathbf{v}_{ij}) \mathbf{e}_{ij}
         * 
         * @f]
         * 
         * The random force is: 
         * 
         * @f[
         * 
         *      \mathbf{F}^R_{ij} = \sigma \left(1 - \frac{r_{ij}}{r_c}\right) \xi_{ij}\Delta t^{-1/2}\mathbf{e}_{ij}
         * 
         * @f]
         * 
         * @param alpha interaction strength of the conservative force. Defaults to 1.0. 
         * @param gamma interaction strength of dissapative force. Defaults to 1.0. 
         * @param sigma strength of random force. Defaults to 1.0. 
         * @param cutoff cutoff distance. Defaults to 1.0. 
         * @param shifted Flag for whether using a shifted potential. Defaults to false. 
         * @return Potential* 
         */
        static Potential *dpd(FPTYPE *alpha=NULL, FPTYPE *gamma=NULL, FPTYPE *sigma=NULL, FPTYPE *cutoff=NULL, bool *shifted=NULL);

        /**
         * @brief Creates a custom potential. 
         * 
         * @param min The smallest radius for which the potential will be constructed.
         * @param max The largest radius for which the potential will be constructed.
         * @param f function returning the value of the potential
         * @param fp function returning the value of first derivative of the potential
         * @param f6p function returning the value of sixth derivative of the potential
         * @param tol Tolerance, defaults to 0.001.
         * @return Potential* 
         */
        static Potential *custom(
            FPTYPE min, 
            FPTYPE max, 
            FPTYPE (*f)(FPTYPE), 
            FPTYPE (*fp)(FPTYPE), 
            FPTYPE (*f6p)(FPTYPE), 
            FPTYPE *tol=NULL, 
            uint32_t *flags=NULL
        );

        FPTYPE getMin();
        FPTYPE getMax();
        FPTYPE getCutoff();
        std::pair<FPTYPE, FPTYPE> getDomain();
        int getIntervals();
        bool getBound();
        void setBound(const bool &_bound);
        FPTYPE getR0();
        void setR0(const FPTYPE &_r0);
        bool getShifted();
        bool getPeriodic();
        bool getRSquare();

    };


    /** Fictitious null potential. */
    CAPI_DATA(struct Potential) potential_null;


    /* associated functions */

    /**
     * @brief Free the memory associated with the given potential.
     * 
     * @param p Pointer to the #potential to clear.
     */
    CAPI_FUNC(void) potential_clear(struct Potential *p);

    /**
     * @brief Construct a #potential from the given function.
     *
     * @param p A pointer to an empty #potential.
     * @param f A pointer to the potential function to be interpolated.
     * @param fp A pointer to the first derivative of @c f.
     * @param f6p A pointer to the sixth derivative of @c f.
     * @param a The smallest radius for which the potential will be constructed.
     * @param b The largest radius for which the potential will be constructed.
     * @param tol The absolute tolerance to which the interpolation should match
     *      the exact potential.
     *
     * Computes an interpolated potential function from @c f in @c [a,b] to the
     * locally relative tolerance @c tol.
     *
     * The sixth derivative @c f6p is used to compute the optimal node
     * distribution. If @c f6p is @c NULL, the derivative is approximated
     * numerically.
     *
     * The zeroth interval contains a linear extension of @c f for values < a.
     */
    CAPI_FUNC(HRESULT) potential_init(
        struct Potential *p, 
        FPTYPE (*f)(FPTYPE),
        FPTYPE (*fp)(FPTYPE), 
        FPTYPE (*f6p)(FPTYPE),
        FPTYPE a, 
        FPTYPE b, 
        FPTYPE tol
    );

    CAPI_FUNC(HRESULT) potential_getcoeffs(
        FPTYPE (*f)(FPTYPE), 
        FPTYPE (*fp)(FPTYPE),
        FPTYPE *xi, 
        int n, 
        FPTYPE *c, 
        FPTYPE *err
    );

    CAPI_FUNC(FPTYPE) potential_getalpha(FPTYPE (*f6p)(FPTYPE), FPTYPE a, FPTYPE b);

};

#endif // _MDCORE_INCLUDE_TFPOTENTIAL_H_