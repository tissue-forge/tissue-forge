/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

/* include some standard header files */
#include <tfPotential.h>
#include <tfDPDPotential.h>
#include <tfParticle.h>
#include <tf_util.h>
#include <tfLogger.h>
#include <tfError.h>
#include <io/tfFIO.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstring>

/* include local headers */
#include <tf_errs.h>
#include <tf_fptype.h>
#include "tf_potential_eval.h"
#include <tf_mdcore_io.h>

#include <iostream>
#include <cmath>


using namespace TissueForge;


#define potential_defarg(ptr, defval) ptr == nullptr ? defval : *ptr

/** Macro to easily define vector types. */
#define simd_vector(elcount, type)  __attribute__((vector_size((elcount)*sizeof(type)))) type

/** The last error */
int TissueForge::potential_err = potential_err_ok;

FPTYPE c_null[] = { FPTYPE_ZERO, FPTYPE_ZERO, FPTYPE_ZERO, FPTYPE_ZERO, FPTYPE_ZERO, FPTYPE_ZERO, FPTYPE_ZERO, FPTYPE_ZERO };

TissueForge::Potential::Potential() : 
    kind(POTENTIAL_KIND_POTENTIAL), 
    flags(POTENTIAL_NONE), 
    alpha{FPTYPE_ZERO, FPTYPE_ZERO, FPTYPE_ZERO, FPTYPE_ZERO}, 
    c(c_null), 
    r0_plusone(1.0), 
    a(0.0f), 
    b(std::numeric_limits<FPTYPE>::max()), 
    mu(0.0), 
	offset{FPTYPE_ZERO, FPTYPE_ZERO, FPTYPE_ZERO}, 
    n(1), 
    create_func(NULL), 
	clear_func(NULL), 
    eval_byparts(NULL), 
    eval_byparts3(NULL), 
    eval_byparts4(NULL), 
	pca(NULL), 
	pcb(NULL), 
    name(NULL)
{}


/** The null potential */
Potential TissueForge::potential_null;

/* the error macro. */
#define error(id)(potential_err = errs_register(id, potential_err_msg[-(id)], __LINE__, __FUNCTION__, __FILE__))

/* list of error messages. */
const char *potential_err_msg[] = {
		"Nothing bad happened.",
		"An unexpected NULL pointer was encountered.",
		"A call to malloc failed, probably due to insufficient memory.",
		"The requested value was out of bounds.",
		"Not yet implemented.",
		"Maximum number of intervals reached before tolerance satisfied."
};

static Potential *potential_checkerr(Potential *p) {
    if(p == NULL) {
        std::string err = errs_getstring(0);
        throw std::runtime_error(err);
    }
    return p;
}

/**
 * @brief Switching function.
 *
 * @param r The radius.
 * @param A The start of the switching region.
 * @param B The end of the switching region.
 */
FPTYPE TissueForge::potential_switch(FPTYPE r, FPTYPE A, FPTYPE B) {

	if(r < A)
		return 1.0;
	else if(r > B)
		return 0.0;
	else {

		FPTYPE B2 = B*B, A2 = A*A, r2 = r*r;
		FPTYPE B2mr2 = B2 - r2, B2mA2 = B2 - A2;

		return B2mr2*B2mr2 *(B2 + 2*r2 - 3*A2) /(B2mA2 * B2mA2 * B2mA2);

	}

}

FPTYPE TissueForge::potential_switch_p(FPTYPE r, FPTYPE A, FPTYPE B) {

	if(A < r && r < B) {

		FPTYPE B2 = B*B, A2 = A*A, r2 = r*r;
		FPTYPE B2mr2 = B2 - r2, B2mA2 = B2 - A2;
		FPTYPE r2_p = 2*r, B2mr2_p = -r2_p;

		return(2*B2mr2_p*B2mr2 *(B2 + 2*r2 - 3*A2) + B2mr2*B2mr2 * 2*r2_p) /(B2mA2 * B2mA2 * B2mA2);

	}
	else
		return 0.0;

}

static void potential_scale(Potential *p) {
	p->flags |= POTENTIAL_SCALED;
}

static void potential_shift(Potential *p, const FPTYPE &r0) {
	p->flags |= POTENTIAL_SHIFTED;
	p->r0_plusone = r0 + 1;
}

static Potential *_do_create_periodic_potential(Potential* (*potCtor)(FPTYPE, FPTYPE, unsigned int), 
												  const unsigned int &order, 
												  const unsigned int dim, 
												  const FPTYPE &max, 
												  const bool &pos) 
{
	TF_Log(LOG_TRACE);

	FPTYPE imageLen = FPTYPE(order) * _Engine.s.dim[dim];
	FPTYPE imageMin = imageLen - max;
	FPTYPE imageMax = imageLen + max;

	FPTYPE offset[3] = {0.0, 0.0, 0.0};
	if (pos) offset[dim] = imageLen;
	else offset[dim] = -imageLen;
	
	Potential *p = (*potCtor)(imageMin, imageMax, order);
	for (int k = 0; k < 3; k++) p->offset[k] = offset[k];
	p->flags |= POTENTIAL_PERIODIC;
	return p;
}

static Potential* create_periodic_potential(Potential* (*potCtor)(FPTYPE, FPTYPE, unsigned int), const unsigned int &order, const FPTYPE &min, const FPTYPE &max) {
	Potential *p = (*potCtor)(min, max, 0);
	if (order == 0) return p;

	Potential *pp;

	std::vector<unsigned int> dims;
	if (_Engine.boundary_conditions.periodic & space_periodic_x) dims.push_back(0);
	if (_Engine.boundary_conditions.periodic & space_periodic_y) dims.push_back(1);
	if (_Engine.boundary_conditions.periodic & space_periodic_z) dims.push_back(2);

	for (unsigned int orderIdx = 1; orderIdx <= order; ++orderIdx) {
		for (auto d : dims) {
			pp = new Potential(*p);
			*p = *pp + *_do_create_periodic_potential(potCtor, orderIdx, d, max, true);
			pp = new Potential(*p);
			*p = *pp + *_do_create_periodic_potential(potCtor, orderIdx, d, max, false);
		}
	}

	return p;
}


/**
 * @brief A basic 12-6 Lennard-Jones potential.
 *
 * @param r The interaction radius.
 * @param A First parameter of the potential.
 * @param B Second parameter of the potential.
 *
 * @return The potential @f$ \left(\frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$
 *      evaluated at @c r.
 */
FPTYPE TissueForge::potential_LJ126(FPTYPE r, FPTYPE A, FPTYPE B) {

	FPTYPE ir = 1.0/r, ir2 = ir * ir, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;

	return(A * ir12 - B * ir6);

}

/**
 * @brief A basic 12-6 Lennard-Jones potential (first derivative).
 *
 * @param r The interaction radius.
 * @param A First parameter of the potential.
 * @param B Second parameter of the potential.
 *
 * @return The first derivative of the potential
 *      @f$ \left(\frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$
 *      evaluated at @c r.
 */
FPTYPE TissueForge::potential_LJ126_p(FPTYPE r, FPTYPE A, FPTYPE B) {

	FPTYPE ir = 1.0/r, ir2 = ir*ir, ir4 = ir2*ir2, ir12 = ir4*ir4*ir4;

	return 6.0 * ir *(-2.0 * A * ir12 + B * ir4 * ir2);

}

/**
 * @brief A basic 12-6 Lennard-Jones potential (sixth derivative).
 *
 * @param r The interaction radius.
 * @param A First parameter of the potential.
 * @param B Second parameter of the potential.
 *
 * @return The sixth derivative of the potential
 *      @f$ \left(\frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$
 *      evaluated at @c r.
 */
FPTYPE TissueForge::potential_LJ126_6p(FPTYPE r, FPTYPE A, FPTYPE B) {

	FPTYPE r2 = r * r, ir2 = 1.0 / r2, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;

	return 10080.0 * ir12 *(884.0 * A * ir6 - 33.0 * B);

}

/**
 * @brief The Coulomb potential.
 *
 * @param r The interaction radius.
 *
 * @return The potential @f$ \frac{1}{4\pi r} @f$
 *      evaluated at @c r.
 */
FPTYPE TissueForge::potential_Coulomb(FPTYPE r) {

	return potential_escale / r;

}

/**
 * @brief The Coulomb potential (first derivative).
 *
 * @param r The interaction radius.
 *
 * @return The first derivative of the potential @f$ \frac{1}{4\pi r} @f$
 *      evaluated at @c r.
 */
FPTYPE TissueForge::potential_Coulomb_p(FPTYPE r) {

	return -potential_escale / (r*r);

}

/**
 * @brief TheCoulomb potential (sixth derivative).
 *
 * @param r The interaction radius.
 *
 * @return The sixth derivative of the potential @f$ \frac{1}{4\pi r} @f$
 *      evaluated at @c r.
 */
FPTYPE TissueForge::potential_Coulomb_6p(FPTYPE r) {

	FPTYPE r2 = r*r, r4 = r2*r2, r7 = r*r2*r4;

	return 720.0 * potential_escale / r7;

}


/**
 * @brief The short-range part of an Ewald summation.
 *
 * @param r The interaction radius.
 * @param kappa The screening length of the Ewald summation.
 *
 * @return The potential @f$ \frac{\mbox{erfc}(\kappa r)}{r} @f$
 *      evaluated at @c r.
 */
FPTYPE TissueForge::potential_Ewald(FPTYPE r, FPTYPE kappa) {

	return potential_escale * erfc(kappa * r) / r;

}

/**
 * @brief The short-range part of an Ewald summation (first derivative).
 *
 * @param r The interaction radius.
 * @param kappa The screening length of the Ewald summation.
 *
 * @return The first derivative of the potential @f$ \frac{\mbox{erfc}(\kappa r)}{r} @f$
 *      evaluated at @c r.
 */
FPTYPE TissueForge::potential_Ewald_p(FPTYPE r, FPTYPE kappa) {

	FPTYPE r2 = r*r, ir = 1.0 / r, ir2 = ir*ir;
	const FPTYPE isqrtpi = 0.56418958354775628695;

	return potential_escale *(-2.0 * exp(-kappa*kappa * r2) * kappa * ir * isqrtpi -
			erfc(kappa * r) * ir2);

}

/**
 * @brief The short-range part of an Ewald summation (sixth derivative).
 *
 * @param r The interaction radius.
 * @param kappa The screening length of the Ewald summation.
 *
 * @return The sixth derivative of the potential @f$ \frac{\mbox{erfc}(\kappa r)}{r} @f$
 *      evaluated at @c r.
 */
FPTYPE TissueForge::potential_Ewald_6p(FPTYPE r, FPTYPE kappa) {

	FPTYPE r2 = r*r, ir2 = 1.0 / r2, r4 = r2*r2, ir4 = ir2*ir2, ir6 = ir2*ir4;
	FPTYPE kappa2 = kappa*kappa;
	FPTYPE t6, t23;
	const FPTYPE isqrtpi = 0.56418958354775628695;

	t6 = erfc(kappa*r);
	t23 = exp(-kappa2*r2);
	return potential_escale *(720.0*t6/r*ir6+(1440.0*ir6+(960.0*ir4+(384.0*ir2+(144.0+(-128.0*r2+64.0*kappa2*r4)*kappa2)*kappa2)*kappa2)*kappa2)*kappa*isqrtpi*t23);

}


FPTYPE potential_create_harmonic_K;
FPTYPE potential_create_harmonic_r0;

/* the potential functions */
FPTYPE potential_create_harmonic_f(FPTYPE r) {
	return potential_create_harmonic_K *(r - potential_create_harmonic_r0) *(r - potential_create_harmonic_r0);
}

FPTYPE potential_create_harmonic_dfdr(FPTYPE r) {
	return 2.0 * potential_create_harmonic_K *(r - potential_create_harmonic_r0);
}

FPTYPE potential_create_harmonic_d6fdr6(FPTYPE r) {
	return 0;
}

/**
 * @brief Creates a harmonic bond #potential
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param K The energy of the bond.
 * @param r0 The minimum energy distance.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ K(r-r_0)^2 @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */
struct Potential *TissueForge::potential_create_harmonic(FPTYPE a, FPTYPE b, FPTYPE K, FPTYPE r0, FPTYPE tol) {

	struct Potential *p = new Potential();

    p->flags = POTENTIAL_HARMONIC & POTENTIAL_R2 ;
    p->name = "Harmonic";

	/* fill this potential */
	potential_create_harmonic_K = K;
	potential_create_harmonic_r0 = r0;
	if(potential_init(p,
                        &potential_create_harmonic_f,
                        &potential_create_harmonic_dfdr,
                        &potential_create_harmonic_d6fdr6,
                        a, b, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
    return p;

}



FPTYPE potential_create_linear_k;

/* the potential functions */
FPTYPE potential_create_linear_f(FPTYPE r) {
    return potential_create_linear_k * r;
}

FPTYPE potential_create_linear_dfdr(FPTYPE r) {
    return potential_create_linear_k;
}

FPTYPE potential_create_linear_d6fdr6(FPTYPE r) {
    return 0;
}

/**
 * @brief Creates a harmonic bond #potential
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param K The energy of the bond.
 * @param r0 The minimum energy distance.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ K(r-r_0)^2 @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */
struct Potential *TissueForge::potential_create_linear (FPTYPE a, FPTYPE b,
                                             FPTYPE k,
                                             FPTYPE tol) {
    
    struct Potential *p = new Potential();
    
    p->flags =  POTENTIAL_R2 ;
    p->name = "Linear";
    
    /* fill this potential */
    potential_create_linear_k = k;
    if(potential_init(p, &potential_create_linear_f, NULL, &potential_create_linear_d6fdr6, a, b, tol) < 0) {
        aligned_Free(p);
        return NULL;
    }
    
    /* return it */
    return p;
}


FPTYPE potential_create_cosine_dihedral_K;
int potential_create_cosine_dihedral_n;
FPTYPE potential_create_cosine_dihedral_delta;

/* the potential functions */
FPTYPE potential_create_cosine_dihedral_f(FPTYPE r) {
	FPTYPE T[potential_create_cosine_dihedral_n+1], U[potential_create_cosine_dihedral_n+1];
	FPTYPE cosd = cos(potential_create_cosine_dihedral_delta), sind = sin(potential_create_cosine_dihedral_delta);
	int k;
	T[0] = 1.0; T[1] = r;
	U[0] = 1.0; U[1] = 2*r;
	for(k = 2 ; k <= potential_create_cosine_dihedral_n ; k++) {
		T[k] = 2 * r * T[k-1] - T[k-2];
		U[k] = 2 * r * U[k-1] - U[k-2];
	}
	if(potential_create_cosine_dihedral_delta == 0.0)
		return potential_create_cosine_dihedral_K *(1.0 + T[potential_create_cosine_dihedral_n]);
	else if(potential_create_cosine_dihedral_delta == M_PI)
		return potential_create_cosine_dihedral_K *(1.0 - T[potential_create_cosine_dihedral_n]);
	else if(fabs(r) < 1.0)
		return potential_create_cosine_dihedral_K *(1.0 + T[potential_create_cosine_dihedral_n]*cosd + U[potential_create_cosine_dihedral_n-1]*sqrt(1.0-r*r)*sind);
	else
		return potential_create_cosine_dihedral_K *(1.0 + T[potential_create_cosine_dihedral_n]*cosd);
}

FPTYPE potential_create_cosine_dihedral_dfdr(FPTYPE r) {
	FPTYPE T[potential_create_cosine_dihedral_n+1], U[potential_create_cosine_dihedral_n+1];
	FPTYPE cosd = cos(potential_create_cosine_dihedral_delta), sind = sin(potential_create_cosine_dihedral_delta);
	int k;
	T[0] = 1.0; T[1] = r;
	U[0] = 1.0; U[1] = 2*r;
	for(k = 2 ; k <= potential_create_cosine_dihedral_n ; k++) {
		T[k] = 2 * r * T[k-1] - T[k-2];
		U[k] = 2 * r * U[k-1] - U[k-2];
	}
	if(potential_create_cosine_dihedral_delta == 0.0)
		return potential_create_cosine_dihedral_n * potential_create_cosine_dihedral_K * potential_create_cosine_dihedral_n*U[potential_create_cosine_dihedral_n-1];
	else if(potential_create_cosine_dihedral_delta == M_PI)
		return -potential_create_cosine_dihedral_n * potential_create_cosine_dihedral_K * potential_create_cosine_dihedral_n*U[potential_create_cosine_dihedral_n-1];
	else
		return potential_create_cosine_dihedral_n * potential_create_cosine_dihedral_K * (U[potential_create_cosine_dihedral_n-1] * cosd - T[potential_create_cosine_dihedral_n] * sind / sqrt(1-r*r));
}

FPTYPE potential_create_cosine_dihedral_d6fdr6(FPTYPE r) {
	return 0.0;
}

/**
 * @brief Creates a harmonic dihedral #potential
 *
 * @param K The energy of the dihedral.
 * @param n The multiplicity of the dihedral.
 * @param delta The minimum energy dihedral.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ K(1 + \cos(n\arccos(r)-delta) @f$ in @f$[-1,1]@f$
 *      or @c NULL on error (see #potential_err).
 */
struct Potential *TissueForge::potential_create_cosine_dihedral(FPTYPE K, int n, FPTYPE delta, FPTYPE tol) {

	struct Potential *p = new Potential();
	FPTYPE a = -1.0, b = 1.0;

	/* Adjust end-points if delta is not a multiple of pi. */
	if(fmod(delta, M_PI) != 0) {
		a = -1.0 / (1.0 - sqrt(FPTYPE_EPSILON));
		b = 1.0 / (1.0 - sqrt(FPTYPE_EPSILON));
	}

    p->flags =   POTENTIAL_R | POTENTIAL_DIHEDRAL;
    p->name = "Cosine Dihedral";

	/* fill this potential */
	potential_create_cosine_dihedral_K = K;
	potential_create_cosine_dihedral_n = n;
	potential_create_cosine_dihedral_delta = delta;
	if(potential_init(p, &potential_create_cosine_dihedral_f, NULL, &potential_create_cosine_dihedral_d6fdr6, a, b, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
			return p;

}


FPTYPE potential_create_harmonic_angle_K;
FPTYPE potential_create_harmonic_angle_theta0;

/* the potential functions */
FPTYPE potential_create_harmonic_angle_f(FPTYPE r) {
	FPTYPE theta;
	r = fmin(1.0, fmax(-1.0, r));
	theta = acos(r);
	return potential_create_harmonic_angle_K *(theta - potential_create_harmonic_angle_theta0) *(theta - potential_create_harmonic_angle_theta0);
}

FPTYPE potential_create_harmonic_angle_dfdr(FPTYPE r) {
	FPTYPE r2 = r*r;
	if(r2 == 1.0)
		return -2.0 * potential_create_harmonic_angle_K;
	else
		return -2.0 * potential_create_harmonic_angle_K *(acos(r) - potential_create_harmonic_angle_theta0) / sqrt(1.0 - r2);
}

FPTYPE potential_create_harmonic_angle_d6fdr6(FPTYPE r) {
	return 0.0;
}

/**
 * @brief Creates a harmonic angle #potential
 *
 * @param a The smallest angle for which the potential will be constructed.
 * @param b The largest angle for which the potential will be constructed.
 * @param K The energy of the angle.
 * @param theta0 The minimum energy angle.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ K(\arccos(r)-r_0)^2 @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */
struct Potential *TissueForge::potential_create_harmonic_angle(FPTYPE a, FPTYPE b, FPTYPE K, FPTYPE theta0, FPTYPE tol) {

	struct Potential *p = new Potential();
	FPTYPE left, right;

    p->flags = POTENTIAL_ANGLE | POTENTIAL_HARMONIC ;
    p->name = "Harmonic Angle";

	/* Adjust a and b accordingly. */
	if(a < 0.0)
		a = 0.0;
	if(b > M_PI)
		b = M_PI;
	left = cos(b);
	right = cos(a);
	
    // the potential_init will automatically padd these already.
    //if(left - fabs(left)*sqrt(FPTYPE_EPSILON) < -1.0)
	//	left = -1.0 /(1.0 + sqrt(FPTYPE_EPSILON));
	//if(right + fabs(right)*sqrt(FPTYPE_EPSILON) > 1.0)
	//	right = 1.0 /(1.0 + sqrt(FPTYPE_EPSILON));

	/* fill this potential */
	potential_create_harmonic_angle_K = K;
	potential_create_harmonic_angle_theta0 = theta0;
	if(potential_init(p, &potential_create_harmonic_angle_f, NULL, &potential_create_harmonic_angle_d6fdr6, left, right, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
	return p;

}


FPTYPE potential_create_harmonic_dihedral_K;
FPTYPE potential_create_harmonic_dihedral_delta;

/* the potential functions */
FPTYPE potential_create_harmonic_dihedral_f(FPTYPE r) {
	FPTYPE theta;
	r = fmin(1.0, fmax(-1.0, r));
	theta = acos(r);
	return potential_create_harmonic_dihedral_K *(theta - potential_create_harmonic_dihedral_delta) *(theta - potential_create_harmonic_dihedral_delta);
}

FPTYPE potential_create_harmonic_dihedral_dfdr(FPTYPE r) {
	FPTYPE r2 = r*r;
	if(r2 == 1.0)
		return -2.0 * potential_create_harmonic_dihedral_K;
	else
		return -2.0 * potential_create_harmonic_dihedral_K *(acos(r) - potential_create_harmonic_dihedral_delta) / sqrt(1.0 - r2);
}

FPTYPE potential_create_harmonic_dihedral_d6fdr6(FPTYPE r) {
	return 0.0;
}

/**
 * @brief Creates a harmonic dihedral #potential
 *
 * @param a The smallest angle for which the potential will be constructed.
 * @param b The largest angle for which the potential will be constructed.
 * @param K The energy of the angle.
 * @param delta The minimum energy angle.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ K(\theta - \delta)^2 @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */
struct Potential *TissueForge::potential_create_harmonic_dihedral(FPTYPE a, FPTYPE b, FPTYPE K, FPTYPE delta, FPTYPE tol) {

	struct Potential *p = new Potential();
	FPTYPE left, right;

    p->flags = POTENTIAL_DIHEDRAL | POTENTIAL_HARMONIC ;
    p->name = "Harmonic Dihedral";

	/* Adjust a and b accordingly. */
	if(a < 0.0)
		a = 0.0;
	if(b > M_PI)
		b = M_PI;
	left = cos(b);
	right = cos(a);

	/* fill this potential */
	potential_create_harmonic_dihedral_K = K;
	potential_create_harmonic_dihedral_delta = delta;
	if(potential_init(p, &potential_create_harmonic_dihedral_f, NULL, &potential_create_harmonic_dihedral_d6fdr6, left, right, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
	return p;

}


FPTYPE potential_create_Ewald_q;
FPTYPE potential_create_Ewald_kappa;

/* the potential functions */
FPTYPE potential_create_Ewald_f(FPTYPE r) {
	return potential_create_Ewald_q * potential_Ewald(r, potential_create_Ewald_kappa);
}

FPTYPE potential_create_Ewald_dfdr(FPTYPE r) {
	return potential_create_Ewald_q * potential_Ewald_p(r, potential_create_Ewald_kappa);
}

FPTYPE potential_create_Ewald_d6fdr6(FPTYPE r) {
	return potential_create_Ewald_q * potential_Ewald_6p(r, potential_create_Ewald_kappa);
}

/**
 * @brief Creates a #potential representing the real-space part of an Ewald 
 *      potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param q The charge scaling of the potential.
 * @param kappa The screening distance of the Ewald potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ q\frac{\mbox{erfc}(\kappa r}{r} @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */
struct Potential *TissueForge::potential_create_Ewald(FPTYPE a, FPTYPE b, FPTYPE q, FPTYPE kappa, FPTYPE tol) {

	struct Potential *p = new Potential();

    p->flags =  POTENTIAL_R2 | POTENTIAL_EWALD ;
    p->name = "Ewald";

	/* fill this potential */
	potential_create_Ewald_q = q;
	potential_create_Ewald_kappa = kappa;
	if(potential_init(p, &potential_create_Ewald_f, &potential_create_Ewald_dfdr, &potential_create_Ewald_d6fdr6, a, b, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
			return p;

}

FPTYPE potential_create_Ewald_periodic_q;
FPTYPE potential_create_Ewald_periodic_kappa;
FPTYPE potential_create_Ewald_periodic_tol;

Potential *_do_potential_create_Ewald_periodic(FPTYPE a, FPTYPE b, unsigned int order) {
	return potential_create_Ewald(a, b, potential_create_Ewald_periodic_q, potential_create_Ewald_periodic_kappa, potential_create_Ewald_periodic_tol);
}

Potential *potential_create_Ewald_periodic(const FPTYPE &a, const FPTYPE &b, const FPTYPE &q, const FPTYPE &kappa, const FPTYPE &tol, const unsigned int &order) {
	potential_create_Ewald_periodic_q = q;
	potential_create_Ewald_periodic_kappa = kappa;
	potential_create_Ewald_periodic_tol = tol;

	return create_periodic_potential(&_do_potential_create_Ewald_periodic, order, a, b);
}

FPTYPE potential_create_LJ126_Ewald_A;
FPTYPE potential_create_LJ126_Ewald_B;
FPTYPE potential_create_LJ126_Ewald_kappa;
FPTYPE potential_create_LJ126_Ewald_q;

/* the potential functions */
FPTYPE potential_create_LJ126_Ewald_f(FPTYPE r) {
	return potential_LJ126(r, potential_create_LJ126_Ewald_A, potential_create_LJ126_Ewald_B) +
			potential_create_LJ126_Ewald_q * potential_Ewald(r, potential_create_LJ126_Ewald_kappa);
}

FPTYPE potential_create_LJ126_Ewald_dfdr(FPTYPE r) {
	return potential_LJ126_p(r, potential_create_LJ126_Ewald_A, potential_create_LJ126_Ewald_B) +
			potential_create_LJ126_Ewald_q * potential_Ewald_p(r, potential_create_LJ126_Ewald_kappa);
}

FPTYPE potential_create_LJ126_Ewald_d6fdr6(FPTYPE r) {
	return potential_LJ126_6p(r, potential_create_LJ126_Ewald_A, potential_create_LJ126_Ewald_B) +
			potential_create_LJ126_Ewald_q * potential_Ewald_6p(r, potential_create_LJ126_Ewald_kappa);
}

/**
 * @brief Creates a #potential representing the sum of a
 *      12-6 Lennard-Jones potential and the real-space part of an Ewald 
 *      potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param q The charge scaling of the potential.
 * @param kappa The screening distance of the Ewald potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left(\frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */
struct Potential *TissueForge::potential_create_LJ126_Ewald(FPTYPE a, FPTYPE b, FPTYPE A, FPTYPE B, FPTYPE q, FPTYPE kappa, FPTYPE tol) {

	struct Potential *p = new Potential();

    p->flags =  POTENTIAL_R2 | POTENTIAL_LJ126 |  POTENTIAL_EWALD ;
    p->name = "Lennard-Jones Ewald";

	/* fill this potential */
	potential_create_LJ126_Ewald_A = A;
	potential_create_LJ126_Ewald_B = B;
	potential_create_LJ126_Ewald_kappa = kappa;
	potential_create_LJ126_Ewald_q = q;
	if(potential_init(p, &potential_create_LJ126_Ewald_f, &potential_create_LJ126_Ewald_dfdr, &potential_create_LJ126_Ewald_d6fdr6, a, b, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
			return p;

}


FPTYPE potential_create_LJ126_Ewald_switch_A;
FPTYPE potential_create_LJ126_Ewald_switch_B;
FPTYPE potential_create_LJ126_Ewald_switch_kappa;
FPTYPE potential_create_LJ126_Ewald_switch_q;
FPTYPE potential_create_LJ126_Ewald_switch_s;
FPTYPE potential_create_LJ126_Ewald_switch_cutoff;

/* the potential functions */
FPTYPE potential_create_LJ126_Ewald_switch_f(FPTYPE r) {
	return potential_LJ126(r, potential_create_LJ126_Ewald_switch_A, potential_create_LJ126_Ewald_switch_B) * potential_switch(r, potential_create_LJ126_Ewald_switch_s, potential_create_LJ126_Ewald_switch_cutoff) +
			potential_create_LJ126_Ewald_switch_q * potential_Ewald(r, potential_create_LJ126_Ewald_switch_kappa);
}

FPTYPE potential_create_LJ126_Ewald_switch_dfdr(FPTYPE r) {
	return potential_LJ126_p(r, potential_create_LJ126_Ewald_switch_A, potential_create_LJ126_Ewald_switch_B) * potential_switch(r, potential_create_LJ126_Ewald_switch_s, potential_create_LJ126_Ewald_switch_cutoff) +
			potential_LJ126(r, potential_create_LJ126_Ewald_switch_A, potential_create_LJ126_Ewald_switch_B) * potential_switch_p(r, potential_create_LJ126_Ewald_switch_s, potential_create_LJ126_Ewald_switch_cutoff) +
			potential_create_LJ126_Ewald_switch_q * potential_Ewald_p(r, potential_create_LJ126_Ewald_switch_kappa);
}

FPTYPE potential_create_LJ126_Ewald_switch_d6fdr6(FPTYPE r) {
	return potential_LJ126_6p(r, potential_create_LJ126_Ewald_switch_A, potential_create_LJ126_Ewald_switch_B) +
			potential_create_LJ126_Ewald_switch_q * potential_Ewald_6p(r, potential_create_LJ126_Ewald_switch_kappa);
}

/**
 * @brief Creates a #potential representing the sum of a
 *      12-6 Lennard-Jones potential with a switching distance
 *      and the real-space part of an Ewald potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param q The charge scaling of the potential.
 * @param s The switching distance.
 * @param kappa The screening distance of the Ewald potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left(\frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */
struct Potential *TissueForge::potential_create_LJ126_Ewald_switch(FPTYPE a, FPTYPE b, FPTYPE A, FPTYPE B, FPTYPE q, FPTYPE kappa, FPTYPE s, FPTYPE tol) {

	struct Potential *p = new Potential();

    p->flags =  POTENTIAL_R2 | POTENTIAL_LJ126 | POTENTIAL_EWALD | POTENTIAL_SWITCH ;
    p->name = "Lennard-Jones Ewald Switch";

	/* fill this potential */
	potential_create_LJ126_Ewald_switch_A = A;
	potential_create_LJ126_Ewald_switch_B = B;
	potential_create_LJ126_Ewald_switch_kappa = kappa;
	potential_create_LJ126_Ewald_switch_q = q;
	potential_create_LJ126_Ewald_switch_s = s;
	potential_create_LJ126_Ewald_switch_cutoff = b;
	if(potential_init(p, &potential_create_LJ126_Ewald_switch_f, &potential_create_LJ126_Ewald_switch_dfdr, &potential_create_LJ126_Ewald_switch_d6fdr6, a, b, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
			return p;

}


FPTYPE potential_create_Coulomb_q;
FPTYPE potential_create_Coulomb_b;

/* the potential functions */
FPTYPE potential_create_Coulomb_f(FPTYPE r) {
	return potential_escale * potential_create_Coulomb_q *(1.0/r - 1.0/potential_create_Coulomb_b);
}

FPTYPE potential_create_Coulomb_dfdr(FPTYPE r) {
	return -potential_escale * potential_create_Coulomb_q /(r * r);
}

FPTYPE potential_create_Coulomb_d6fdr6(FPTYPE r) {
	FPTYPE r2 = r*r, r4 = r2*r2, r7 = r*r2*r4;
	return 720.0 * potential_escale * potential_create_Coulomb_q / r7;
}

/**
 * @brief Creates a #potential representing a shifted Coulomb potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param q The charge scaling of the potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \frac{1}{4\pi r} @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */
struct Potential *TissueForge::potential_create_Coulomb(FPTYPE a, FPTYPE b, FPTYPE q, FPTYPE tol) {

	struct Potential *p = new Potential();

    p->flags =  POTENTIAL_R2 |  POTENTIAL_COULOMB ;
    p->name = "Coulomb";

	/* fill this potential */
	potential_create_Coulomb_q = q;
	potential_create_Coulomb_b = b;
	if(potential_init(p, &potential_create_Coulomb_f, &potential_create_Coulomb_dfdr, &potential_create_Coulomb_d6fdr6, a, b, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
			return p;

}

FPTYPE potential_create_Coulomb_periodic_q;
FPTYPE potential_create_Coulomb_periodic_tol;

Potential *_do_potential_create_Coulomb_periodic(FPTYPE min, FPTYPE max, unsigned int order) {
	return potential_create_Coulomb(min, max, potential_create_Coulomb_periodic_q, potential_create_Coulomb_periodic_tol);
}

Potential *potential_create_Coulomb_periodic(FPTYPE a, FPTYPE b, FPTYPE q, FPTYPE tol, const unsigned int &order) {
	potential_create_Coulomb_periodic_q = q;
	potential_create_Coulomb_periodic_tol = tol;
	return create_periodic_potential(&_do_potential_create_Coulomb_periodic, order, a, b);
}


FPTYPE potential_create_LJ126_Coulomb_q;
FPTYPE potential_create_LJ126_Coulomb_b;
FPTYPE potential_create_LJ126_Coulomb_A;
FPTYPE potential_create_LJ126_Coulomb_B;

/* the potential functions */
FPTYPE potential_create_LJ126_Coulomb_f(FPTYPE r) {
	return potential_LJ126(r, potential_create_LJ126_Coulomb_A, potential_create_LJ126_Coulomb_B) +
			potential_escale * potential_create_LJ126_Coulomb_q *(1.0/r - 1.0/potential_create_LJ126_Coulomb_b);
}

FPTYPE potential_create_LJ126_Coulomb_dfdr(FPTYPE r) {
	return potential_LJ126_p(r, potential_create_LJ126_Coulomb_A, potential_create_LJ126_Coulomb_B) -
			potential_escale * potential_create_LJ126_Coulomb_q /(r * r);
}

FPTYPE potential_create_LJ126_Coulomb_d6fdr6(FPTYPE r) {
	FPTYPE r2 = r*r, r4 = r2*r2, r7 = r*r2*r4;
	return potential_LJ126_6p(r, potential_create_LJ126_Coulomb_A, potential_create_LJ126_Coulomb_B) +
			720.0 * potential_escale * potential_create_LJ126_Coulomb_q / r7;
}

/**
 * @brief Creates a #potential representing the sum of a
 *      12-6 Lennard-Jones potential and a shifted Coulomb potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param q The charge scaling of the potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left(\frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */
struct Potential *TissueForge::potential_create_LJ126_Coulomb(FPTYPE a, FPTYPE b, FPTYPE A, FPTYPE B, FPTYPE q, FPTYPE tol) {

	struct Potential *p = new Potential();

    p->flags =  POTENTIAL_R2 | POTENTIAL_COULOMB | POTENTIAL_LJ126  ;
    p->name = "Lennard-Jones Coulomb";

	/* fill this potential */
	potential_create_LJ126_Coulomb_q = q;
	potential_create_LJ126_Coulomb_b = b;
	potential_create_LJ126_Coulomb_A = A;
	potential_create_LJ126_Coulomb_B = B;
	if(potential_init(p, &potential_create_LJ126_Coulomb_f, &potential_create_LJ126_Coulomb_dfdr, &potential_create_LJ126_Coulomb_d6fdr6, a, b, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
			return p;

}


FPTYPE potential_create_LJ126_A;
FPTYPE potential_create_LJ126_B;

/* the potential functions */
FPTYPE potential_create_LJ126_f(FPTYPE r) {
	return potential_LJ126(r, potential_create_LJ126_A, potential_create_LJ126_B);
}

FPTYPE potential_create_LJ126_dfdr(FPTYPE r) {
	return potential_LJ126_p(r, potential_create_LJ126_A, potential_create_LJ126_B);
}

FPTYPE potential_create_LJ126_d6fdr6(FPTYPE r) {
	return potential_LJ126_6p(r, potential_create_LJ126_A, potential_create_LJ126_B);
}

/**
 * @brief Creates a #potential representing a 12-6 Lennard-Jones potential
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left(\frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 *
 */
struct Potential *TissueForge::potential_create_LJ126(FPTYPE a, FPTYPE b, FPTYPE A, FPTYPE B, FPTYPE tol) {

    Potential *p = new Potential();

    p->flags =  POTENTIAL_R2  | POTENTIAL_LJ126 ;
    p->name = "Lennard-Jones";

	/* fill this potential */
	potential_create_LJ126_A = A;
	potential_create_LJ126_B = B;
	if(potential_init(p, &potential_create_LJ126_f, &potential_create_LJ126_dfdr, &potential_create_LJ126_d6fdr6, a, b, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
	return p;

}


FPTYPE potential_create_LJ126_switch_A;
FPTYPE potential_create_LJ126_switch_B;
FPTYPE potential_create_LJ126_switch_s;
FPTYPE potential_create_LJ126_switch_cutoff;

/* the potential functions */
FPTYPE potential_create_LJ126_switch_f(FPTYPE r) {
	return potential_LJ126(r, potential_create_LJ126_switch_A, potential_create_LJ126_switch_B) * potential_switch(r, potential_create_LJ126_switch_s, potential_create_LJ126_switch_cutoff);
}

FPTYPE potential_create_LJ126_switch_dfdr(FPTYPE r) {
	return potential_LJ126_p(r, potential_create_LJ126_switch_A, potential_create_LJ126_switch_B) * potential_switch(r, potential_create_LJ126_switch_s, potential_create_LJ126_switch_cutoff) +
			potential_LJ126(r, potential_create_LJ126_switch_A, potential_create_LJ126_switch_B) * potential_switch_p(r, potential_create_LJ126_switch_s, potential_create_LJ126_switch_cutoff);
}

FPTYPE potential_create_LJ126_switch_d6fdr6(FPTYPE r) {
	return potential_LJ126_6p(r, potential_create_LJ126_switch_A, potential_create_LJ126_switch_B);
}

/**
 * @brief Creates a #potential representing a switched 12-6 Lennard-Jones potential
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param s The switchting length
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left(\frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 *
 */
struct Potential *TissueForge::potential_create_LJ126_switch(FPTYPE a, FPTYPE b, FPTYPE A, FPTYPE B, FPTYPE s, FPTYPE tol) {

	struct Potential *p = new Potential();

    p->flags =  POTENTIAL_R2 | POTENTIAL_LJ126 | POTENTIAL_SWITCH ;
    p->name = "Lennard-Jones Switch";

	/* fill this potential */
	potential_create_LJ126_switch_A = A;
	potential_create_LJ126_switch_B = B;
	potential_create_LJ126_switch_s = s;
	potential_create_LJ126_switch_cutoff = b;
	if(potential_init(p, &potential_create_LJ126_switch_f, &potential_create_LJ126_switch_dfdr, &potential_create_LJ126_switch_d6fdr6, a, b, tol) < 0) {
		aligned_Free(p);
		return NULL;
	}

	/* return it */
			return p;

}



//#define Power(base, exp) std::pow(base, exp)

FPTYPE Power(FPTYPE base, FPTYPE exp) {
    FPTYPE result = std::pow(base, exp);
    return result;
}

#define MLog(x) std::log(x)

/**
 * @brief Free the memory associated with the given potential.
 * 
 * @param p Pointer to the #potential to clear.
 */
void TissueForge::potential_clear(struct Potential *p) {

	/* Do nothing? */
	if(p == NULL)
		return;

	/* Issue callback */
	if(p->clear_func) 
		p->clear_func(p);
	
	/* Clear the flags. */
	p->flags = POTENTIAL_NONE;

	/* Clear the coefficients. */
	aligned_Free(p->c);
	p->c = NULL;

}


static FPTYPE overlapping_sphere_k;
static FPTYPE overlapping_sphere_mu;
static FPTYPE overlapping_sphere_k_harmonic;
static FPTYPE overlapping_sphere_harmonic_r0;
static FPTYPE overlapping_sphere_harmonic_k;

// overlapping sphere f
//Piecewise(List(List(-x + x*MLog(x),x <= 1)),-1 + Power(k,-2) - (Power(E,k - k*x)*(1 + k*(-1 + x)))/Power(k,2)),
static FPTYPE overlapping_sphere_f(FPTYPE x) {
    FPTYPE k = overlapping_sphere_k;
    FPTYPE mu = overlapping_sphere_mu;
    FPTYPE harmonic_r0 = overlapping_sphere_harmonic_r0;
    FPTYPE harmonic_k = overlapping_sphere_harmonic_k;
    
    FPTYPE result;
    if(x <= 1) {
        result =  -x + x*MLog(x);
    }
    else {
        result =  -1 + Power(k,-2) - (Power(M_E,k - k*x)*(1 + k*(-1 + x)))/Power(k,2);
    }
    
    //Log(LOG_TRACE) << "fdata = Append[fdata, {" << x << ", " << result << "}];";
    return mu * result + harmonic_k*Power(-x + harmonic_r0,2);
    //return harmonic_k*Power(-x + harmonic_r0,2);
}

// overlapping sphere fp
//Piecewise(List(List(MLog(x),x < 1),List(Power(E,k*(1 - x))*(-1 + x),x >= 1)),0),
static FPTYPE overlapping_sphere_fp(FPTYPE x) {
    FPTYPE k = overlapping_sphere_k;
    FPTYPE mu = overlapping_sphere_mu;
    FPTYPE harmonic_r0 = overlapping_sphere_harmonic_r0;
    FPTYPE harmonic_k = overlapping_sphere_harmonic_k;
    FPTYPE result;
    if(x <= 1) {
        result = MLog(x);
    }
    else {
        result =  Power(M_E,k*(1 - x))*(-1 + x);
    }
    
    //Log(LOG_TRACE) << "fpdata = Append[fpdata, {" << x << ", " << result << "}];";
    return  mu * result + 2*harmonic_k*(-x + harmonic_r0);
    //return 2*harmonic_k*(-x + harmonic_r0);
}

// overlapping sphere f6p
//Piecewise(List(List(24/Power(x,5),x < 1),List(Power(E,k - k*x)*Power(k,4) -
//    Power(E,k - k*x)*Power(k,4)*(-4 - k + k*x),x > 1)),Indeterminate)
static FPTYPE overlapping_sphere_f6p(FPTYPE x) {
    
    FPTYPE k = overlapping_sphere_k;
    FPTYPE mu = overlapping_sphere_mu;
    FPTYPE result;
    if(x <= 1) {
        result =  24/Power(x,5);
    }
    else {
        result =  Power(M_E,k - k*x)*Power(k,4) - Power(M_E,k - k*x)*Power(k,4)*(-4 - k + k*x);
    }
    //Log(LOG_TRACE) << "fp6data = Append[fp6data, {" << x << ", " << result << "}];";
    return mu * result;
    //return 0;
}

struct Potential *potential_create_overlapping_sphere(FPTYPE mu, FPTYPE k,
    FPTYPE harmonic_k, FPTYPE harmonic_r0,
    FPTYPE a, FPTYPE b,FPTYPE tol) {
    
    struct Potential *p = new Potential();
    
    overlapping_sphere_mu = mu;
    overlapping_sphere_k = k;
    overlapping_sphere_harmonic_k = harmonic_k;
    overlapping_sphere_harmonic_r0 = harmonic_r0;
    
    
    p->flags =  POTENTIAL_R2  | POTENTIAL_LJ126;
    
    if(harmonic_k == 0.0) {
        p->name = "Overlapping Sphere";
    }
    else {
        p->name = "Overlapping Sphere with Harmonic";
    }
    
    int err = 0;
    
    if((err = potential_init(p,&overlapping_sphere_f,
                             &overlapping_sphere_fp,
                             &overlapping_sphere_f6p, a, b, tol)) < 0) {
        
        TF_Log(LOG_ERROR) << "error creating potential: " << potential_err_msg[-err];
        aligned_Free(p);
        return NULL;
    }
	
	potential_scale(p);
    
    /* return it */
    return p;
}

static FPTYPE power_k;
static FPTYPE power_r0;
static FPTYPE power_alpha;

static FPTYPE power_f(FPTYPE x) {
    FPTYPE k = power_k;
    FPTYPE r0 = power_r0;
    FPTYPE alpha = power_alpha;
    
    FPTYPE result;
    
    result =  k * Power(std::abs(-r0 + x), alpha);
    
    return result;
}

static FPTYPE power_fp(FPTYPE x) {
    FPTYPE k = power_k;
    FPTYPE r0 = power_r0;
    FPTYPE alpha = power_alpha;
    FPTYPE result;
    
    result = alpha * k + Power(std::abs(-r0 + x), -1 + alpha);
    
    return result;
}

static FPTYPE power_f6p(FPTYPE x) {
    
    FPTYPE k = power_k;
    FPTYPE r0 = power_r0;
    FPTYPE alpha = power_alpha;
    
    FPTYPE result;
    
    result =  (-5 + alpha) *
           (-4 + alpha) *
           (-3 + alpha) *
           (-2 + alpha) *
           (-1 + alpha) *
           alpha * k * Power(std::abs(-r0 + x), -6 + alpha);
    
    return result;
}

struct Potential *potential_create_power(FPTYPE k, FPTYPE r0, FPTYPE alpha, FPTYPE a, FPTYPE b,FPTYPE tol) {
    
    struct Potential *p = new Potential();
    
    power_k = k;
    power_r0 = r0;
    power_alpha = alpha;
    
    p->flags =  POTENTIAL_R2;
    
    p->name = "Power(r - r0)^alpha";
    
    int err = 0;
    
    // interpolate potential bigger than the bounds so we dont get jaggy edges
    FPTYPE min, max;
    FPTYPE range = b - a;
    
    FPTYPE fudge = range / 5;
    
    FPTYPE fudged_a = a;
    
    if(a - fudge >= 0.001) {
        fudged_a = a - fudge;
    }
    
    
    if((err = potential_init(p,&power_f,
                             &power_fp,
                             &power_f6p, fudged_a, 1.2 * b, tol)) < 0) {
        
        TF_Log(LOG_ERROR) << "error creating potential: " << potential_err_msg[-err];
        aligned_Free(p);
        return NULL;
    }
    
	potential_scale(p);
    
    /* return it */
    return p;
}




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
 * @return #potential_err_ok or <0 on error (see #potential_err).
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
int TissueForge::potential_init (struct Potential *p,
                    FPTYPE (*f)(FPTYPE),
                    FPTYPE (*fp)(FPTYPE),
                    FPTYPE (*f6p)(FPTYPE),
                    FPTYPE a, FPTYPE b, FPTYPE tol) {
	TF_Log(LOG_DEBUG);

	FPTYPE alpha, w;
	int l = potential_ivalsa, r = potential_ivalsb, m;
	FPTYPE err_l = 0, err_r = 0, err_m = 0;
	FPTYPE *xi_l = NULL, *xi_r = NULL, *xi_m = NULL;
	FPTYPE *c_l = NULL, *c_r = NULL, *c_m = NULL;
	int i = 0, k = 0;
	FPTYPE e;
	FPTYPE mtol = 10 * FPTYPE_EPSILON;

	/* check inputs */
	if(p == NULL || f == NULL) { 
		TF_Log(LOG_CRITICAL);
		return error(potential_err_null);
	}

	/* check if we have a user-specified 6th derivative or not. */
	if(f6p == NULL) {
		TF_Log(LOG_CRITICAL);
		return error(potential_err_nyi);
	}
    
    /* set the boundaries */
    p->a = a; p->b = b;

	/* Stretch the domain ever so slightly to accommodate for rounding
       error when computing the index. */
	b += fabs(b) * sqrt(FPTYPE_EPSILON);
	a -= fabs(a) * sqrt(FPTYPE_EPSILON);

	

	/* compute the optimal alpha for this potential */
	alpha = potential_getalpha(f6p,a,b);

	/* compute the interval transform */
	w = 1.0 / (a - b); w *= w;
	p->alpha[0] = a*a*w - alpha*b*a*w;
	p->alpha[1] = -2*a*w + alpha*(a+b)*w;
	p->alpha[2] = w - alpha*w;
	p->alpha[3] = 0.0;

	/* Correct the transform to the right. */
	w = 2*FPTYPE_EPSILON*(fabs(p->alpha[0])+fabs(p->alpha[1])+fabs(p->alpha[2]));
	p->alpha[0] -= w*a/(a-b);
	p->alpha[1] += w/(a-b);

	/* compute the smallest interpolation... */
	xi_l = (FPTYPE *)aligned_Malloc(sizeof(FPTYPE) * (l + 1), potential_align);
	c_l = (FPTYPE *)aligned_Malloc(sizeof(FPTYPE) * (l+1) * potential_chunk, potential_align);
	if (xi_l == NULL || c_l == NULL) {
		TF_Log(LOG_CRITICAL);
		return error(potential_err_malloc);
    }
	xi_l[0] = a; xi_l[l] = b;
	for(i = 1 ; i < l ; i++) {
		xi_l[i] = a + (b - a) * i / l;
		while(1) {
			e = i - l * (p->alpha[0] + xi_l[i]*(p->alpha[1] + xi_l[i]*p->alpha[2]));
			xi_l[i] += e / (l * (p->alpha[1] + 2*xi_l[i]*p->alpha[2]));
			if(fabs(e) < l*mtol)
				break;
		}
	}
	if(potential_getcoeffs(f,fp,xi_l,l,&c_l[potential_chunk],&err_l) < 0) { 
		TF_Log(LOG_CRITICAL);
		return error(potential_err);
	}

	/* if this interpolation is good enough, stop here! */
	if(err_l < tol) {
		TF_Log(LOG_DEBUG);

		/* Set the domain variables. */
		p->n = l;
		p->c = c_l;
		p->alpha[0] *= p->n; p->alpha[1] *= p->n; p->alpha[2] *= p->n;
		p->alpha[0] += 1;

		/* Fix the first interval. */
		p->c[0] = a; p->c[1] = 1.0 / a;
		FPTYPE coeffs[potential_degree], eff[potential_degree];
		for(k = 0 ; k < potential_degree ; k++) {
			coeffs[k] = p->c[2*potential_chunk-1-k];
			eff[k] = 0.0;
		}
		for(i = 0 ; i < potential_degree ; i++)
			for(k = potential_degree-1 ; k >= i ; k--) {
				eff[i] = coeffs[k] + (-1.0)*eff[i];
				coeffs[k] *= (k - i) * p->c[potential_chunk+1] * a;
			}
		p->c[potential_chunk-1] = eff[0];
		p->c[potential_chunk-2] = eff[1];
		p->c[potential_chunk-3] = 0.5 * eff[2];
		// p->c[potential_chunk-4] = (eff[2] - eff[1]) / 3;
		for(k = 3 ; k <= potential_degree ; k++)
			p->c[potential_chunk-1-k] = 0.0;

		/* Clean up. */
		aligned_Free(xi_l);
        
        assert(int(FPTYPE_FMAX(FPTYPE_ZERO, p->alpha[0] + p->b * (p->alpha[1] + p->b * p->alpha[2]))) < p->n + 1);
		return potential_err_ok;
	}

	/* loop until we have an upper bound on the right... */
	TF_Log(LOG_DEBUG);
	while(1) {

		/* compute the larger interpolation... */
		xi_r = (FPTYPE*)aligned_Malloc(sizeof(FPTYPE) * (r + 1),  potential_align);
		c_r =  (FPTYPE*)aligned_Malloc(sizeof(FPTYPE) * (r + 1) * potential_chunk, potential_align);
        if(xi_r == NULL || c_r == NULL) {
			TF_Log(LOG_CRITICAL);
			return error(potential_err_malloc);
        }
		xi_r[0] = a; xi_r[r] = b;
		for(i = 1 ; i < r ; i++) {
			xi_r[i] = a + (b - a) * i / r;
			while(1) {
				e = i - r * (p->alpha[0] + xi_r[i]*(p->alpha[1] + xi_r[i]*p->alpha[2]));
				xi_r[i] += e / (r * (p->alpha[1] + 2*xi_r[i]*p->alpha[2]));
				if(fabs(e) < r*mtol)
					break;
			}
		}
        if(potential_getcoeffs(f,fp,xi_r,r,&c_r[potential_chunk],&err_r) < 0) {
			TF_Log(LOG_CRITICAL) << "Error in potential_getcoeffs";
			return error(potential_err);
        }

		/* if this is better than tolerance, break... */
        if(err_r < tol) {
			break;
        }

		/* Have we too many intervals? */
		else if(2*r > potential_ivalsmax) {
			TF_Log(LOG_CRITICAL) << "Too many intervals (" << err_r << ")";
			return error(potential_err_ivalsmax);
		}

		/* otherwise, l=r and r = 2*r */
		else {
			l = r; err_l = err_r;
			aligned_Free(xi_l); xi_l = xi_r;
			aligned_Free(c_l); c_l = c_r;
			r *= 2;
		}

	} /* loop until we have a good right estimate */

	/* we now have a left and right estimate -- binary search! */
	TF_Log(LOG_DEBUG);
	while(r - l > 1) {

		/* find the middle */
		m = 0.5 *(r + l);

		/* construct that interpolation */
		xi_m = (FPTYPE*)aligned_Malloc(sizeof(FPTYPE) * (m + 1), potential_align);
		c_m =  (FPTYPE*)aligned_Malloc(sizeof(FPTYPE) * (m + 1) * potential_chunk, potential_align);

        if(xi_m == NULL || c_m == NULL) {
			TF_Log(LOG_CRITICAL);
			return error(potential_err_malloc);
        }
		xi_m[0] = a; xi_m[m] = b;
		for(i = 1 ; i < m ; i++) {
			xi_m[i] = a + (b - a) * i / m;
			while(1) {
				e = i - m * (p->alpha[0] + xi_m[i]*(p->alpha[1] + xi_m[i]*p->alpha[2]));
				xi_m[i] += e / (m * (p->alpha[1] + 2*xi_m[i]*p->alpha[2]));
				if(fabs(e) < m*mtol)
					break;
			}
		}
		if(potential_getcoeffs(f,fp,xi_m,m,&c_m[potential_chunk],&err_m) != 0) {
			TF_Log(LOG_CRITICAL);
			return error(potential_err);
		}

		/* go left? */
		if(err_m > tol) {
			l = m; err_l = err_m;
			aligned_Free(xi_l); xi_l = xi_m;
			aligned_Free(c_l); c_l = c_m;
		}

		/* otherwise, go right... */
		else {
			r = m; err_r = err_m;
			aligned_Free(xi_r); xi_r = xi_m;
			aligned_Free(c_r); c_r = c_m;
		}

	} /* binary search */

	/* as of here, the right estimate is the smallest interpolation below */
	/* the requested tolerance */
	TF_Log(LOG_DEBUG);
	p->n = r;
	p->c = c_r;
	p->alpha[0] *= p->n; p->alpha[1] *= p->n; p->alpha[2] *= p->n;
	p->alpha[0] += 1.0;

	/* Make the first interval a linear continuation. */
	p->c[0] = a; p->c[1] = 1.0 / a;
	FPTYPE coeffs[potential_degree], eff[potential_degree];
	for(k = 0 ; k < potential_degree ; k++) {
		coeffs[k] = p->c[2*potential_chunk-1-k];
		eff[k] = 0.0;
	}
	for(i = 0 ; i < potential_degree ; i++)
		for(k = potential_degree-1 ; k >= i ; k--) {
			eff[i] = coeffs[k] + (-1.0)*eff[i];
			coeffs[k] *= (k - i) * p->c[potential_chunk+1] * a;
		}
	p->c[potential_chunk-1] = eff[0];
	p->c[potential_chunk-2] = eff[1];
	p->c[potential_chunk-3] = 0.5 * eff[2];
	for(k = 3 ; k <= potential_degree ; k++)
		p->c[potential_chunk-1-k] = 0.0;

	/* Clean up. */
	aligned_Free(xi_r);
	aligned_Free(xi_l);
	aligned_Free(c_l);

	/* all is well that ends well... */
    
	return potential_err_ok;
}


/**
 * @brief Compute the optimal first derivatives for the given set of
 *      nodes.
 *
 * @param f Pointer to the function to be interpolated.
 * @param n Number of intervals.
 * @param xi Pointer to an array of nodes between whicht the function @c f
 *      will be interpolated.
 * @param fp Pointer to an array in which to store the first derivatives
 *      of @c f.
 *
 * @return #potential_err_ok or < 0 on error (see #potential_err).
 */
int potential_getfp(FPTYPE (*f)(FPTYPE), int n, FPTYPE *x, FPTYPE *fp) {

	int i, k;
	FPTYPE m, h, eff, fx[n+1], hx[n];
	FPTYPE d0[n+1], d1[n+1], d2[n+1], b[n+1];
	FPTYPE viwl1[n], viwr1[n];
	static FPTYPE *w = NULL, *xi = NULL;

	/* Cardinal functions. */
	const FPTYPE cwl1[4] = { 0.25, -0.25, -0.25, 0.25 };
	const FPTYPE cwr1[4] = { -0.25, -0.25, 0.25, 0.25 };
	const FPTYPE wl0wl1 = 0.1125317885884428;
	const FPTYPE wl1wl1 = 0.03215579530433858;
	const FPTYPE wl0wr1 = -0.04823369227661384;
	const FPTYPE wl1wr1 = -0.02143719641629633;
	const FPTYPE wr0wr1 = -0.1125317885884429;
	const FPTYPE wr1wr1 = 0.03215579530433859;
	const FPTYPE wl1wr0 = 0.04823369227661384;

	/* Pre-compute the weights? */
	if(w == NULL) {
		if((w = (FPTYPE *)malloc(sizeof(FPTYPE) * potential_N)) == NULL ||
				(xi = (FPTYPE *)malloc(sizeof(FPTYPE) * potential_N)) == NULL)
			return error(potential_err_malloc);
		for(k = 1 ; k < potential_N-1 ; k++) {
			xi[k] = cos(k * M_PI / (potential_N - 1));
			w[k] = 1.0 / sqrt(1.0 - xi[k]*xi[k]);
		}
		xi[0] = 1.0; xi[potential_N-1] = -1.0;
		w[0] = 0.0; w[potential_N-1] = 0.0;
	}

	/* Get the values of fx and ih. */
	for(i = 0 ; i <= n ; i++)
		fx[i] = f(x[i]);
	for(i = 0 ; i < n ; i++)
		hx[i] = x[i+1] - x[i];

	/* Compute the products of f with respect to wl1 and wr1. */
	for(i = 0 ; i < n ; i++) {
		viwl1[i] = 0.0; viwr1[i] = 0.0;
		m = 0.5*(x[i] + x[i+1]);
		h = 0.5*(x[i+1] - x[i]);
		for(k = 1 ; k < potential_N-1 ; k++) {
			eff = f(m + h*xi[k]);
			viwl1[i] += w[k] *(eff *(cwl1[0] + xi[k]*(cwl1[1] + xi[k]*(cwl1[2] + xi[k]*cwl1[3]))));
			viwr1[i] += w[k] *(eff *(cwr1[0] + xi[k]*(cwr1[1] + xi[k]*(cwr1[2] + xi[k]*cwr1[3]))));
		}
		viwl1[i] /= potential_N-2;
		viwr1[i] /= potential_N-2;
	}

	/* Fill the diagonals and the right-hand side. */
	d1[0] = wl1wl1 * hx[0];
	d2[0] = wl1wr1 * hx[0];
	b[0] = 2 *(viwl1[0] - fx[0]*wl0wl1 - fx[1]*wl1wr0);
	for(i = 1 ; i < n ; i++) {
		d0[i] = wl1wr1 * hx[i-1];
		d1[i] = wr1wr1 * hx[i-1] + wl1wl1 * hx[i];
		d2[i] = wl1wr1 * hx[i];
		b[i] = 2 *(viwr1[i-1] - fx[i-1]*wl0wr1 - fx[i]*wr0wr1) +
				2 *(viwl1[i] - fx[i]*wl0wl1 - fx[i+1]*wl1wr0);
	}
	d0[n] = wl1wr1 * hx[n-1];
	d1[n] = wr1wr1 * hx[n-1];
	b[n] = 2 *(viwr1[n-1] - fx[n-1]*wl0wr1 - fx[n]*wr0wr1);

	/* Solve the trilinear system. */
	for(i = 1 ; i <= n ; i++)  {
		m = d0[i]/d1[i-1];
		d1[i] = d1[i] - m*d2[i-1];
		b[i] = b[i] - m*b[i-1];
	}
	fp[n] = b[n]/d1[n];
	for(i = n - 1 ; i >= 0 ; i--)
		fp[i] =(b[i] - d2[i]*fp[i+1]) / d1[i];

	/* Fingers crossed... */
	return potential_err_ok;

}


int potential_getfp_fixend(FPTYPE (*f)(FPTYPE), FPTYPE fpa, FPTYPE fpb, int n, FPTYPE *x, FPTYPE *fp) {

	int i, k;
	FPTYPE m, h, eff, fx[n+1], hx[n];
	FPTYPE d0[n+1], d1[n+1], d2[n+1], b[n+1];
	FPTYPE viwl1[n], viwr1[n];
	static FPTYPE *w = NULL, *xi = NULL;

	/* Cardinal functions. */
	const FPTYPE cwl1[4] = { 0.25, -0.25, -0.25, 0.25 };
	const FPTYPE cwr1[4] = { -0.25, -0.25, 0.25, 0.25 };
	const FPTYPE wl0wl1 = 0.1125317885884428;
	const FPTYPE wl1wl1 = 0.03215579530433858;
	const FPTYPE wl0wr1 = -0.04823369227661384;
	const FPTYPE wl1wr1 = -0.02143719641629633;
	const FPTYPE wr0wr1 = -0.1125317885884429;
	const FPTYPE wr1wr1 = 0.03215579530433859;
	const FPTYPE wl1wr0 = 0.04823369227661384;

	/* Pre-compute the weights? */
	if(w == NULL) {
		if((w = (FPTYPE *)malloc(sizeof(FPTYPE) * potential_N)) == NULL ||
				(xi = (FPTYPE *)malloc(sizeof(FPTYPE) * potential_N)) == NULL)
			return error(potential_err_malloc);
		for(k = 1 ; k < potential_N-1 ; k++) {
			xi[k] = cos(k * M_PI / (potential_N - 1));
			w[k] = 1.0 / sqrt(1.0 - xi[k]*xi[k]);
		}
		xi[0] = 1.0; xi[potential_N-1] = -1.0;
		w[0] = 0.0; w[potential_N-1] = 0.0;
	}

	/* Get the values of fx and ih. */
	for(i = 0 ; i <= n ; i++)
		fx[i] = f(x[i]);
	for(i = 0 ; i < n ; i++)
		hx[i] = x[i+1] - x[i];

	/* Compute the products of f with respect to wl1 and wr1. */
	for(i = 0 ; i < n ; i++) {
		viwl1[i] = 0.0; viwr1[i] = 0.0;
		m = 0.5*(x[i] + x[i+1]);
		h = 0.5*(x[i+1] - x[i]);
		for(k = 1 ; k < potential_N-1 ; k++) {
			eff = f(m + h*xi[k]);
			viwl1[i] += w[k] *(eff *(cwl1[0] + xi[k]*(cwl1[1] + xi[k]*(cwl1[2] + xi[k]*cwl1[3]))));
			viwr1[i] += w[k] *(eff *(cwr1[0] + xi[k]*(cwr1[1] + xi[k]*(cwr1[2] + xi[k]*cwr1[3]))));
		}
		viwl1[i] /= potential_N-2;
		viwr1[i] /= potential_N-2;
	}

	/* Fill the diagonals and the right-hand side. */
	d1[0] = 1.0;
	d2[0] = 0.0;
	b[0] = fpa;
	for(i = 1 ; i < n ; i++) {
		d0[i] = wl1wr1 * hx[i-1];
		d1[i] = wr1wr1 * hx[i-1] + wl1wl1 * hx[i];
		d2[i] = wl1wr1 * hx[i];
		b[i] = 2 *(viwr1[i-1] - fx[i-1]*wl0wr1 - fx[i]*wr0wr1) +
				2 *(viwl1[i] - fx[i]*wl0wl1 - fx[i+1]*wl1wr0);
	}
	d0[n] = 0.0;
	d1[n] = 1.0;
	b[n] = fpb;

	/* Solve the trilinear system. */
	for(i = 1 ; i <= n ; i++)  {
		m = d0[i]/d1[i-1];
		d1[i] = d1[i] - m*d2[i-1];
		b[i] = b[i] - m*b[i-1];
	}
	fp[n] = b[n]/d1[n];
	for(i = n - 1 ; i >= 0 ; i--)
		fp[i] =(b[i] - d2[i]*fp[i+1]) / d1[i];

	/* Fingers crossed... */
	return potential_err_ok;

}


/**
 * @brief Compute the interpolation coefficients over a given set of nodes.
 * 
 * @param f Pointer to the function to be interpolated.
 * @param fp Pointer to the first derivative of @c f.
 * @param xi Pointer to an array of nodes between whicht the function @c f
 *      will be interpolated.
 * @param n Number of nodes in @c xi.
 * @param c Pointer to an array in which to store the interpolation
 *      coefficients.
 * @param err Pointer to a floating-point value in which an approximation of
 *      the interpolation error, relative to the maximum of f in each interval,
 *      is stored.
 *
 * @return #potential_err_ok or < 0 on error (see #potential_err).
 *
 * Compute the coefficients of the function @c f with derivative @c fp
 * over the @c n intervals between the @c xi and store an estimate of the
 * maximum locally relative interpolation error in @c err.
 *
 * The array to which @c c points must be large enough to hold at least
 * #potential_degree x @c n values of type #FPTYPE.
 */
int TissueForge::potential_getcoeffs(FPTYPE (*f)(FPTYPE), FPTYPE (*fp)(FPTYPE), FPTYPE *xi, int n, FPTYPE *c, FPTYPE *err) {

	// TODO, seriously buggy shit here!
	// make sure all arrays are of length n+1
	int i, j, k, ind;
	FPTYPE phi[7], cee[6], fa, fb, dfa, dfb, fix[n+1], fpx[n+1];
	FPTYPE h, m, w, e, err_loc, maxf, x;
	FPTYPE fx[potential_N];
	static FPTYPE *coskx = NULL;

	/* check input sanity */
	if(f == NULL || xi == NULL || err == NULL)
		return error(potential_err_null);

	/* Do we need to init the pre-computed cosines? */
	if(coskx == NULL) {
		if((coskx = (FPTYPE *)malloc(sizeof(FPTYPE) * 7 * potential_N)) == NULL)
			return error(potential_err_malloc);
		for(k = 0 ; k < 7 ; k++)
			for(j = 0 ; j < potential_N ; j++)
				coskx[ k*potential_N + j ] = cos(j * k * M_PI / potential_N);
	}

	/* Get fx and fpx. */
	for(k = 0 ; k <= n ; k++) {
		fix[k] = f(xi[k]);
		// fpx[k] = fp(xi[k]);
	}

	/* Compute the optimal fpx. */
	if(fp == NULL) {
		if(potential_getfp(f, n, xi, fpx) < 0)
			return error(potential_err);
	}
	else {
		if(potential_getfp_fixend(f, fp(xi[0]), fp(xi[n]), n, xi, fpx) < 0)
			return error(potential_err);
	}
	/* for(k = 0 ; k <= n ; k++)
        printf("potential_getcoeffs: fp[%i]=%e, fpx[%i]=%e.\n", k, fp(xi[k]), k, fpx[k]);
    fflush(stdout); getchar(); */

	/* init the maximum interpolation error */
	*err = 0.0;

	/* loop over all intervals... */
	for(i = 0 ; i < n ; i++) {

		/* set the initial index */
		ind = i * (potential_degree + 3);

		/* get the interval centre and width */
		m = (xi[i] + xi[i+1]) / 2;
		h = (xi[i+1] - xi[i]) / 2;

		/* evaluate f and fp at the edges */
		fa = fix[i]; fb = fix[i+1];
		dfa = fpx[i] * h; dfb = fpx[i+1] * h;
		// printf("potential_getcoeffs: xi[i]=%22.16e\n",xi[i]);

		/* compute the coefficients phi of f */
		for(k = 0 ; k < potential_N ; k++)
			fx[k] = f(m + h * cos(k * M_PI / potential_N));
		for(j = 0 ; j < 7 ; j++) {
			phi[j] = (fa + (1-2*(j%2))*fb) / 2;
			for(k = 1 ; k < potential_N ; k++)
				phi[j] += fx[k] * coskx[ j*potential_N + k ];
			phi[j] *= 2.0 / potential_N;
		}

		/* compute the first four coefficients */
		cee[0] = (4*(fa + fb) + dfa - dfb) / 4;
		cee[1] = -(9*(fa - fb) + dfa + dfb) / 16;
		cee[2] = (dfb - dfa) / 8;
		cee[3] = (fa - fb + dfa + dfb) / 16;
		cee[4] = 0.0;
		cee[5] = 0.0;

		/* add the 4th correction... */
		w =(6 *(cee[0] - phi[0]) - 4 *(cee[2] - phi[2]) - phi[4]) /(36 + 16 + 1);
		cee[0] += -6 * w;
		cee[2] += 4 * w;
		cee[4] = -w;

		/* add the 5th correction... */
		w =(2 *(cee[1] - phi[1]) - 3 *(cee[3] - phi[3]) - phi[5]) /(4 + 9 + 1);
		cee[1] += -2 * w;
		cee[3] += 3 * w;
		cee[5] = -w;

		/* convert to monomials on the interval [-1,1] */
		c[ind+7] = cee[0]/2 - cee[2] + cee[4];
		c[ind+6] = cee[1] - 3*cee[3] + 5*cee[5];
		c[ind+5] = 2*cee[2] - 8*cee[4];
		c[ind+4] = 4*cee[3] - 20*cee[5];
		c[ind+3] = 8*cee[4];
		c[ind+2] = 16*cee[5];
		c[ind+1] = 1.0 / h;
		c[ind] = m;

		/* compute a local error estimate (klutzy) */
		maxf = 0.0; err_loc = 0.0;
		for(k = 1 ; k < potential_N ; k++) {
			maxf = fmax(fabs(fx[k]), maxf);
			x = coskx[ potential_N + k ];
			e = fabs(fx[k] - c[ind+7]
								-x *(c[ind+6] +
										x *(c[ind+5] +
												x *(c[ind+4] +
														x *(c[ind+3] +
																x * c[ind+2])))));
			err_loc = fmax(e, err_loc);
		}
		err_loc /= fmax(maxf, 1.0);
		*err = fmax(err_loc, *err);

	}

	/* all is well that ends well... */
	return potential_err_ok;

}


/**
 * @brief Compute the parameter @f$\alpha@f$ for the optimal node distribution.
 *
 * @param f6p Pointer to a function representing the 6th derivative of the
 *      interpoland.
 * @param a Left limit of the interpolation.
 * @param b Right limit of the interpolation.
 *
 * @return The computed value for @f$\alpha@f$.
 *
 * The value @f$\alpha@f$ is computed using Brent's algortihm to 4 decimal
 * digits.
 */
FPTYPE TissueForge::potential_getalpha(FPTYPE (*f6p)(FPTYPE), FPTYPE a, FPTYPE b) {

	FPTYPE xi[potential_N], fx[potential_N];
	int i, j;
	FPTYPE temp;
	FPTYPE alpha[4], fa[4], maxf = 0.0;
	const FPTYPE golden = 2.0 / (1 + sqrt(5));

	/* start by evaluating f6p at the N nodes between 'a' and 'b' */
	for(i = 0 ; i < potential_N ; i++) {
		xi[i] = ((FPTYPE)i + 1) / (potential_N + 1);
		fx[i] = f6p(a + (b-a) * xi[i]);
		maxf = fmax(maxf, fabs(fx[i]));
	}

	/* Trivial? */
	if(maxf == 0.0)
		return 1.0;

	/* set the initial values for alpha */
	alpha[0] = 0; alpha[3] = 2;
	alpha[1] = alpha[3] - 2 * golden; alpha[2] = alpha[0] + 2 * golden;
	for(i = 0 ; i < 4 ; i++) {
		fa[i] = 0.0;
		for(j = 0 ; j < potential_N ; j++) {
			temp = fabs(pow(alpha[i] + 2 * (1 - alpha[i]) * xi[j], -6) * fx[j]);
			if(temp > fa[i])
				fa[i] = temp;
		}
	}

	/* main loop (brent's algorithm) */
			while(alpha[3] - alpha[0] > 1.0e-4) {

				/* go west? */
				if(fa[1] < fa[2]) {
					alpha[3] = alpha[2]; fa[3] = fa[2];
					alpha[2] = alpha[1]; fa[2] = fa[1];
					alpha[1] = alpha[3] - (alpha[3] - alpha[0]) * golden;
					i = 1;
				}

				/* nope, go east... */
				else {
					alpha[0] = alpha[1]; fa[0] = fa[1];
					alpha[1] = alpha[2]; fa[1] = fa[2];
					alpha[2] = alpha[0] + (alpha[3] - alpha[0]) * golden;
					i = 2;
				}

				/* compute the new value */
				fa[i] = 0.0;
				for(j = 0 ; j < potential_N ; j++) {
					temp = fabs(pow(alpha[i] + 2 * (1 - alpha[i]) * xi[j], -6) * fx[j]);
					if(temp > fa[i])
						fa[i] = temp;
				}

			} /* main loop */

	/* return the average */
	return (alpha[0] + alpha[3]) / 2;

}

FPTYPE TissueForge::Potential::operator()(const FPTYPE &r, const FPTYPE &r0) {
    try {
		FPTYPE e = 0;

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				e += (*c)(r, r0);
			}

			return e;
		}

        FPTYPE f = 0;
		FPTYPE _r = r;
		FPTYPE _r0 = r0;
        // if no r args are given, we pull the r0 from the potential,
        // and use the ri, rj to cancel them out.
        if((flags & POTENTIAL_SCALED || flags & POTENTIAL_SHIFTED) && r0 < 0) {
  
            std::cerr << "calling scaled potential without s, sum of particle radii" << std::endl;
            
            _r0 = 1.0f;

        }

		if(this->flags & POTENTIAL_ANGLE || this->flags & POTENTIAL_DIHEDRAL) {
			_r = cos(_r);
			_r0 = cos(_r0);
		}
        
        if(flags & POTENTIAL_R) {
            potential_eval_r(this, _r, &e, &f);
        }
        else {
            potential_eval_ex(this, _r0/2, _r0/2, _r*_r, &e, &f);
        }
        
        TF_Log(LOG_DEBUG) << "potential_eval(" << r << ") : (" << e << "," << f << ")";
        
        return e;
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return 0;
    }
}

FPTYPE TissueForge::Potential::operator()(const std::vector<FPTYPE> &r) {
	try{
		FPTYPE e = 0;

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				e += (*c)(r);
			}

			return e;
		}

		FVector3 rv(r);
		auto rvl = rv.length();
		FPTYPE f[3] = {0.0, 0.0, 0.0};
		this->eval_byparts(this, NULL, NULL, rv.data(), rvl * rvl, &e, f);
		
        TF_Log(LOG_DEBUG) << "potential_eval(" << rv << ")";

		return e;
	}
	catch (const std::exception& e) {
		tf_exp(e);
		return 0.0;
	}
}

FPTYPE TissueForge::Potential::operator()(ParticleHandle* pi, const FVector3 &pt) {
	try{
		FPTYPE e = 0;

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				e += (*c)(pi, pt);
			}

			return e;
		}

		auto part_i = pi->part();
		FVector3 rv = pi->relativePosition(pt);

		auto rvl = rv.length();
		FPTYPE f[3] = {0.0, 0.0, 0.0};
		this->eval_bypart(this, part_i, rv.data(), rvl * rvl, &e, f);
		
        TF_Log(LOG_DEBUG) << "potential_eval(" << rv << ")";

		return e;
	}
	catch (const std::exception& e) {
		tf_exp(e);
		return 0.0;
	}
}

FPTYPE TissueForge::Potential::operator()(ParticleHandle* pi, ParticleHandle* pj) {
	try{
		FPTYPE e = 0;

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				e += (*c)(pi, pj);
			}

			return e;
		}

		auto part_i = pi->part();
		auto part_j = pj->part();
		FVector3 rv = pi->relativePosition(part_j->position);

		auto rvl = rv.length();
		FPTYPE f[3] = {0.0, 0.0, 0.0};
		this->eval_byparts(this, part_i, part_j, rv.data(), rvl * rvl, &e, f);
		
        TF_Log(LOG_DEBUG) << "potential_eval(" << rv << ")";

		return e;
	}
	catch (const std::exception& e) {
		tf_exp(e);
		return 0.0;
	}
}

FPTYPE TissueForge::Potential::operator()(ParticleHandle* pi, ParticleHandle* pj, ParticleHandle* pk) {
	try{
		FPTYPE e = 0;

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				e += (*c)(pi, pj, pk);
			}

			return e;
		}

		auto part_i = pi->part();
		auto part_j = pj->part();
		auto part_k = pk->part();
		FVector3 rv_ij = pi->relativePosition(part_j->position);
		FVector3 rv_kj = pk->relativePosition(part_j->position);

		FPTYPE ctheta = rv_ij.normalized().dot(rv_kj.normalized());
		FVector3 fi(0.0), fk(0.0);
		this->eval_byparts3(this, part_i, part_j, part_k, ctheta, &e, fi.data(), fk.data());
		
        TF_Log(LOG_DEBUG) << "potential_eval(" << acos(ctheta) << ")";

		return e;
	}
	catch (const std::exception& e) {
		tf_exp(e);
		return 0.0;
	}
}

FPTYPE TissueForge::Potential::operator()(ParticleHandle* pi, ParticleHandle* pj, ParticleHandle* pk, ParticleHandle* pl) {
	try{
		FPTYPE e = 0;

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				e += (*c)(pi, pj, pk, pl);
			}

			return e;
		}

		auto part_i = pi->part();
		auto part_j = pj->part();
		auto part_k = pk->part();
		auto part_l = pl->part();
		
		FQuaternion q_ijk = FQuaternion::fromMatrix(FMatrix3(part_i->position, part_j->position, part_k->position));
		FQuaternion q_jkl = FQuaternion::fromMatrix(FMatrix3(part_j->position, part_k->position, part_l->position));
		FPTYPE phi = q_ijk.angle(q_jkl);

		FVector3 fi(0.0), fl(0.0);
		this->eval_byparts4(this, part_i, part_j, part_k, part_l, cos(phi), &e, fi.data(), fl.data());
		
        TF_Log(LOG_DEBUG) << "potential_eval(" << phi << ")";

		return e;
	}
	catch (const std::exception& e) {
		tf_exp(e);
		return 0.0;
	}
}

FPTYPE TissueForge::Potential::force(FPTYPE r, FPTYPE ri, FPTYPE rj) {
    try {
        FPTYPE f = 0;

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				f += c->force(r, ri, rj);
			}

			return f;
		}
		
        FPTYPE e = 0;
		FPTYPE _r = r;

		if(this->flags & POTENTIAL_ANGLE || this->flags & POTENTIAL_DIHEDRAL) {
			_r = cos(_r);
		}

        // if no r args are given, we pull the r0 from the potential,
        // and use the ri, rj to cancel them out.
        if((flags & POTENTIAL_SHIFTED) && ri < 0 && rj < 0) {
            ri = 1 / 2;
            rj = 1 / 2;
        }
        
        if(flags & POTENTIAL_R) {
            potential_eval_r(this, r, &e, &f);
        }
        else {
            potential_eval_ex(this, ri, rj, r*r, &e, &f);
        }
		
        TF_Log(LOG_DEBUG) << "force_eval(" << r << ")";
        
        return (f * r) / 2;
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return -1.0;
    }
}

std::vector<FPTYPE> TissueForge::Potential::force(const std::vector<FPTYPE>& r) {
	try{
		FVector3 f(0.0);

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				f += c->force(r);
			}

			return std::vector<FPTYPE>(f);
		}

		FVector3 rv(r);
		auto rvl = rv.length();
		FPTYPE e = 0;
		this->eval_byparts(this, NULL, NULL, rv.data(), rvl * rvl, &e, f.data());
		
        TF_Log(LOG_DEBUG) << "force_eval(" << rv << ")";

		return std::vector<FPTYPE>(f);
	}
	catch (const std::exception& e) {
		tf_exp(e);
		return std::vector<FPTYPE>(3, 0.0);
	}
}

std::vector<FPTYPE> TissueForge::Potential::force(ParticleHandle* pi, const FVector3 &pt) {
	try{
		FVector3 f(0.0);

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				f += c->force(pi, pt);
			}

			return std::vector<FPTYPE>(f);
		}

		auto part_i = pi->part();
		FVector3 rv = pi->relativePosition(pt);

		auto rvl = rv.length();
		FPTYPE e = 0;
		this->eval_bypart(this, part_i, rv.data(), rvl * rvl, &e, f.data());
		
        TF_Log(LOG_DEBUG) << "force_eval(" << rv << ")";

		return std::vector<FPTYPE>(f);
	}
	catch (const std::exception& e) {
		tf_exp(e);
		return {0.0, 0.0, 0.0};
	}
}

std::vector<FPTYPE> TissueForge::Potential::force(ParticleHandle* pi, ParticleHandle* pj) {
	try{
		FVector3 f(0.0);

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				f += c->force(pi, pj);
			}

			return std::vector<FPTYPE>(f);
		}

		auto part_i = pi->part();
		auto part_j = pj->part();
		FVector3 rv = pi->relativePosition(part_j->position);

		auto rvl = rv.length();
		FPTYPE e = 0;
		this->eval_byparts(this, part_i, part_j, rv.data(), rvl * rvl, &e, f.data());
		
        TF_Log(LOG_DEBUG) << "force_eval(" << rv << ")";

		return std::vector<FPTYPE>(f);
	}
	catch (const std::exception& e) {
		tf_exp(e);
		return {0.0, 0.0, 0.0};
	}
}

std::pair<std::vector<FPTYPE>, std::vector<FPTYPE> > TissueForge::Potential::force(ParticleHandle* pi, ParticleHandle* pj, ParticleHandle* pk) {
	try{
		FVector3 fi(0.0), fk(0.0);

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				auto fp = c->force(pi, pj, pk);
				fi += std::get<0>(fp);
				fk += std::get<1>(fp);
			}

			return {std::vector<FPTYPE>(fi), std::vector<FPTYPE>(fk)};
		}

		auto part_i = pi->part();
		auto part_j = pj->part();
		auto part_k = pk->part();
		FVector3 rv_ij = pi->relativePosition(part_j->position);
		FVector3 rv_kj = pk->relativePosition(part_j->position);

		FPTYPE ctheta = rv_ij.normalized().dot(rv_kj.normalized());
		FPTYPE e = 0;
		this->eval_byparts3(this, part_i, part_j, part_k, ctheta, &e, fi.data(), fk.data());
		
        TF_Log(LOG_DEBUG) << "force_eval(" << acos(ctheta) << ")";

		return {std::vector<FPTYPE>(fi), std::vector<FPTYPE>(fk)};
	}
	catch (const std::exception& e) {
		tf_exp(e);
		return {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
	}
}

std::pair<std::vector<FPTYPE>, std::vector<FPTYPE> > TissueForge::Potential::force(ParticleHandle* pi, ParticleHandle* pj, ParticleHandle* pk, ParticleHandle* pl) {
	try{
		FVector3 fi(0.0), fl(0.0);

		if(this->kind == POTENTIAL_KIND_COMBINATION) {
			for (auto c : this->constituents()) {
				auto fp = c->force(pi, pj, pk, pl);
				fi += std::get<0>(fp);
				fl += std::get<1>(fp);
			}

			return {std::vector<FPTYPE>(fi), std::vector<FPTYPE>(fl)};
		}

		auto part_i = pi->part();
		auto part_j = pj->part();
		auto part_k = pk->part();
		auto part_l = pl->part();
		
		FQuaternion q_ijk = FQuaternion::fromMatrix(FMatrix3(part_i->position, part_j->position, part_k->position));
		FQuaternion q_jkl = FQuaternion::fromMatrix(FMatrix3(part_j->position, part_k->position, part_l->position));
		FPTYPE phi = q_ijk.angle(q_jkl);

		FPTYPE e = 0;
		this->eval_byparts4(this, part_i, part_j, part_k, part_l, cos(phi), &e, fi.data(), fl.data());
		
        TF_Log(LOG_DEBUG) << "force_eval(" << phi << ")";

		return {std::vector<FPTYPE>(fi), std::vector<FPTYPE>(fl)};
	}
	catch (const std::exception& e) {
		tf_exp(e);
		return {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
	}
}

std::vector<Potential*> TissueForge::Potential::constituents() {
	std::vector<Potential*> result;

	if(pca) {
		if(pca->kind == POTENTIAL_KIND_COMBINATION) {
			auto pcs = pca->constituents();
			result.reserve(result.size() + std::distance(pcs.begin(), pcs.end()));
			result.insert(result.end(), pcs.begin(), pcs.end());
		}
		else result.push_back(pca);
	}
	if(pcb) {
		if(pcb->kind == POTENTIAL_KIND_COMBINATION) {
			auto pcs = pcb->constituents();
			result.reserve(result.size() + std::distance(pcs.begin(), pcs.end()));
			result.insert(result.end(), pcs.begin(), pcs.end());
		}
		else result.push_back(pcb);
	}

	return result;
}

Potential& TissueForge::Potential::operator+(const Potential& rhs) {
	Potential *p = new Potential();

	// Enforce compatibility
	if(this->flags & POTENTIAL_ANGLE && !(rhs.flags & POTENTIAL_ANGLE)) {
		tf_exp(std::invalid_argument("Incompatible potentials"));
		return potential_null;
	}

	p->pca = this;
	p->pcb = const_cast<Potential*>(&rhs);
	p->kind = POTENTIAL_KIND_COMBINATION;
	p->flags = p->flags | POTENTIAL_SUM;

	std::string pName = this->name;
	pName += std::string(" PLUS ");
	pName += p->pcb->name;
	char *cname = new char[pName.size() + 1];
	std::strcpy(cname, pName.c_str());
	p->name = cname;
	return *p;
}

std::string TissueForge::Potential::toString() {
	io::IOElement *fe = new io::IOElement();
    io::MetaData metaData;
    if(io::toFile(this, metaData, fe) != S_OK) 
        return "";
    return io::toStr(fe, metaData);
}

Potential *TissueForge::Potential::fromString(const std::string &str) {
    return io::fromString<Potential*>(str);
}

Potential *TissueForge::Potential::lennard_jones_12_6(FPTYPE min, FPTYPE max, FPTYPE A, FPTYPE B, FPTYPE *tol) {
    TF_Log(LOG_TRACE);

    try {
        return potential_checkerr(potential_create_LJ126(min, max, A, B, potential_defarg(tol, 0.001 * (max - min))));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::lennard_jones_12_6_coulomb(FPTYPE min, FPTYPE max, FPTYPE A, FPTYPE B, FPTYPE q, FPTYPE *tol) {
    TF_Log(LOG_TRACE);

    try {
        return potential_checkerr(potential_create_LJ126_Coulomb(min, max, A, B, q, potential_defarg(tol, 0.001 * (max - min))));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::ewald(FPTYPE min, FPTYPE max, FPTYPE q, FPTYPE kappa, FPTYPE *tol, unsigned int *periodicOrder) {
    TF_Log(LOG_TRACE);

    try {
		unsigned int _periodicOrder = potential_defarg(periodicOrder, 0);
		Potential *p;

		if (_periodicOrder == 0) p = potential_create_Ewald(min, max, q, kappa, potential_defarg(tol, 0.001 * (max - min)));
		else p = potential_create_Ewald_periodic(min, max, q, kappa, potential_defarg(tol, 0.001 * (max - min)), _periodicOrder);

        return potential_checkerr(p);
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::coulomb(FPTYPE q, FPTYPE *min, FPTYPE *max, FPTYPE *tol, unsigned int *periodicOrder) {
    TF_Log(LOG_TRACE);

    try {
		auto _min = potential_defarg(min, 0.01);
		auto _max = potential_defarg(max, 2.0);
		auto _tol = potential_defarg(tol, 0.001 * (_max - _min));
		unsigned int _periodicOrder = potential_defarg(periodicOrder, 0);
		Potential *p;

		if (_periodicOrder == 0) p = potential_create_Coulomb(_min, _max, q, _tol);
		else p = potential_create_Coulomb_periodic(_min, _max, q, _tol, _periodicOrder);

        return potential_checkerr(p);
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

FPTYPE _coulombR_cos_f(FPTYPE r) {
	return cos(r);
}

FPTYPE _coulombR_cos_fp(FPTYPE r) {
	return - sin(r);
}

FPTYPE _coulombR_cos_f6p(FPTYPE r) {
	return - cos(r);
}

FPTYPE _coulombR_sin_f(FPTYPE r) {
	return sin(r);
}

FPTYPE _coulombR_sin_fp(FPTYPE r) {
	return cos(r);
}

FPTYPE _coulombR_sin_f6p(FPTYPE r) {
	return - sin(r);
}

static void coulombR_eval(struct Potential *p, 
						  struct Particle *part_i, 
						  struct Particle *part_j, 
						  FPTYPE *dx, 
						  FPTYPE r2, 
						  FPTYPE *e, 
						  FPTYPE *f);

struct CoulombRPotential : public Potential {
	Potential *pc, *ps;
	unsigned int modes;
	FPTYPE q, kappa;
	std::vector<FPTYPE> mode_cfs;
	std::vector<FVector3> mode_rvecs;

	CoulombRPotential(Potential* pc, 
						Potential* ps, 
						FPTYPE min, 
						FPTYPE max, 
						FPTYPE q, 
						FPTYPE kappa, 
						unsigned int modes, 
						std::vector<FPTYPE> mode_cfs, 
						std::vector<FVector3> mode_rvecs) : 
		Potential(), pc(pc), ps(ps), q(q), kappa(kappa), modes(modes), mode_cfs(mode_cfs), mode_rvecs(mode_rvecs)
	{
		this->a = min;
		this->b = max;
		this->flags |= POTENTIAL_COULOMBR;
		this->kind = POTENTIAL_KIND_BYPARTICLES;
		this->name = "CoulombR";
		this->eval_byparts = &coulombR_eval;
	}
};

void coulombR_eval(struct Potential *p, 
				   struct Particle *part_i, 
				   struct Particle *part_j, 
				   FPTYPE *dx, 
				   FPTYPE r2, 
				   FPTYPE *e, 
				   FPTYPE *f) 
{
	CoulombRPotential *pr = (CoulombRPotential*)p;

	FPTYPE xe, xf, *ce, *cf, vale, valf;
	int inde, indf, k;
	FVector3 fr(0.0);
	FPTYPE TWO_PI = 2.0 * M_PI;

	for (unsigned int m = 0; m < pr->mode_cfs.size(); m++) {
		auto mode_cf = pr->mode_cfs[m];
		auto mode_rvec = pr->mode_rvecs[m];
		FPTYPE r = mode_rvec[0] * dx[0] + mode_rvec[1] * dx[1] + mode_rvec[2] * dx[2];
		FPTYPE r0 = r;
		
		if (r < 0.0) while (r < 0.0) r += TWO_PI;
		else if (r > TWO_PI) while (r > TWO_PI) r -= TWO_PI;
		
		inde = FPTYPE_FMAX(FPTYPE_ZERO, pr->pc->alpha[0] + r * (pr->pc->alpha[1] + r * pr->pc->alpha[2]));
		indf = FPTYPE_FMAX(FPTYPE_ZERO, pr->ps->alpha[0] + r * (pr->ps->alpha[1] + r * pr->ps->alpha[2]));
		ce = &(pr->pc->c[inde * potential_chunk]);
		cf = &(pr->ps->c[indf * potential_chunk]);
		xe = (r - ce[0]) * ce[1];
		xf = (r - cf[0]) * cf[1];
		vale = ce[2] * xe + ce[3];
		valf = cf[2] * xf + cf[3];
		for (k = 4; k < potential_chunk; k++) {
			vale = vale * xe + ce[k];
			valf = valf * xf + cf[k];
		}

		*e += mode_cf * vale;
		fr += mode_cf * valf * mode_rvec;
	}

	f[0] += fr[0];
	f[1] += fr[1];
	f[2] += fr[2];
}

Potential* TissueForge::Potential::coulombR(FPTYPE q, FPTYPE kappa, FPTYPE min, FPTYPE max, unsigned int* modes) {
	unsigned int _modes = potential_defarg(modes, 1);

	std::vector<FPTYPE> mode_cfs;
	std::vector<FVector3> mode_rvecs;

	unsigned int modesX = _Engine.boundary_conditions.periodic & space_periodic_x ? _modes : 0;
	unsigned int modesY = _Engine.boundary_conditions.periodic & space_periodic_y ? _modes : 0;
	unsigned int modesZ = _Engine.boundary_conditions.periodic & space_periodic_z ? _modes : 0;

	FPTYPE A = 2.0 * M_PI / (_Engine.s.dim[0] * _Engine.s.dim[1] * _Engine.s.dim[2]);
	for (unsigned int mx = 0; mx <= modesX; mx++) {
		FPTYPE kx = 2.0 * M_PI * mx / _Engine.s.dim[0];

		for (unsigned int my = 0; my <= modesY; my++) {
			FPTYPE ky = 2.0 * M_PI * my / _Engine.s.dim[1];

			for (unsigned int mz = 0; mz <= modesZ; mz++) {
				FPTYPE kz = 2.0 * M_PI * mz / _Engine.s.dim[2];
				FPTYPE kmag2 = kx * kx + ky * ky + kz * kz;

				if (kmag2 == 0.0) continue;

				mode_rvecs.push_back({kx, ky, kz});

				FPTYPE expPow = M_PI / kappa;
				mode_cfs.push_back(A / kmag2 * q * exp(- expPow * expPow * kmag2));
			}
		}
	}

	Potential* pc = new Potential();
	Potential* ps = new Potential();

	potential_init(pc, &_coulombR_cos_f, &_coulombR_cos_fp, &_coulombR_cos_f6p, 0, 2*M_PI, 1e-4);
	potential_init(ps, &_coulombR_sin_f, &_coulombR_sin_fp, &_coulombR_sin_f6p, 0, 2*M_PI, 1e-4);

	return new CoulombRPotential(pc, ps, min, max, q, kappa, _modes, mode_cfs, mode_rvecs);
}

Potential *TissueForge::Potential::harmonic(FPTYPE k, FPTYPE r0, FPTYPE *min, FPTYPE *max, FPTYPE *tol) {
    TF_Log(LOG_TRACE);

    try {
        auto range = r0;

        auto _min = potential_defarg(min, r0 - range);
        auto _max = potential_defarg(max, r0 + range);
        return potential_checkerr(potential_create_harmonic(_min, _max, k, r0, potential_defarg(tol, 0.01 * (_max - _min))));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::linear(FPTYPE k, FPTYPE *min, FPTYPE *max, FPTYPE *tol) {
    TF_Log(LOG_TRACE);

    try {
		auto _min = potential_defarg(min, std::numeric_limits<FPTYPE>::epsilon());
		auto _max = potential_defarg(max, 10.0);
        return potential_checkerr(potential_create_linear(_min, _max, k, potential_defarg(tol, 0.01 * (_max - _min))));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::harmonic_angle(FPTYPE k, FPTYPE theta0, FPTYPE *min, FPTYPE *max, FPTYPE *tol) {
    TF_Log(LOG_TRACE);

    try {
		auto _min = potential_defarg(min, 0.0);
		auto _max = potential_defarg(max, M_PI);
        return potential_checkerr(potential_create_harmonic_angle(_min, _max, k, theta0, potential_defarg(tol, 0.005 * (_max - _min))));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::harmonic_dihedral(FPTYPE k, FPTYPE delta, FPTYPE *min, FPTYPE *max, FPTYPE *tol) {
    TF_Log(LOG_TRACE);

    try {
		auto _min = potential_defarg(min, 0.0);
		auto _max = potential_defarg(max, M_PI);
        return potential_checkerr(potential_create_harmonic_dihedral(_min, _max, k, delta, potential_defarg(tol, 0.005 * (_max - _min))));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::cosine_dihedral(FPTYPE k, int n, FPTYPE delta, FPTYPE *tol) {
    TF_Log(LOG_TRACE);

    try {
		return potential_checkerr(potential_create_cosine_dihedral(k, n, delta, potential_defarg(tol, 0.01)));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::well(FPTYPE k, FPTYPE n, FPTYPE r0, FPTYPE *min, FPTYPE *max, FPTYPE *tol) {
    TF_Log(LOG_TRACE);

    try {
		auto _min = potential_defarg(min, 0.0);
        auto _max = potential_defarg(max, 0.99 * r0);
        return potential_checkerr(potential_create_well(k, n, r0, 
														potential_defarg(tol, 0.005 * (_max - _min)), 
														_min, _max));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::glj(FPTYPE e, FPTYPE *m, FPTYPE *n, FPTYPE *k, FPTYPE *r0, FPTYPE *min, FPTYPE *max, FPTYPE *tol, bool *shifted) {
    TF_Log(LOG_TRACE);

    try {
        auto _m = potential_defarg(m, 3.0);
		auto _r0 = potential_defarg(r0, 1.0);
		return potential_checkerr(potential_create_glj(e, _m, 
													   potential_defarg(n, 2*_m), 
													   potential_defarg(k, 0.0), 
													   _r0, 
													   potential_defarg(min, 0.05 * _r0), 
													   potential_defarg(max, 3.0 * _r0), 
													   potential_defarg(tol, 0.01), 
													   potential_defarg(shifted, false)));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::morse(FPTYPE *d, FPTYPE *a, FPTYPE *r0, FPTYPE *min, FPTYPE *max, FPTYPE *tol, bool *shifted) {
    TF_Log(LOG_TRACE);

    try {
		return potential_checkerr(potential_create_morse(potential_defarg(d, 1.0), 
														 potential_defarg(a, 6.0), 
														 potential_defarg(r0, 0.0), 
														 potential_defarg(min, 0.0001), 
														 potential_defarg(max, 3.0), 
														 potential_defarg(tol, 0.001), 
														 potential_defarg(shifted, true)));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::overlapping_sphere(FPTYPE *mu, FPTYPE *kc, FPTYPE *kh, FPTYPE *r0, FPTYPE *min, FPTYPE *max, FPTYPE *tol) {
    TF_Log(LOG_TRACE);

    try {
        return potential_checkerr(potential_create_overlapping_sphere(potential_defarg(mu, 1.0), 
																	  potential_defarg(kc, 1.0), 
																	  potential_defarg(kh, 0.0), 
																	  potential_defarg(r0, 0.0), 
																	  potential_defarg(min, 0.001), 
																	  potential_defarg(max, 10.0), 
																	  potential_defarg(tol, 0.001)));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::power(FPTYPE *k, FPTYPE *r0, FPTYPE *alpha, FPTYPE *min, FPTYPE *max, FPTYPE *tol) {
    TF_Log(LOG_TRACE);

    try {
		auto _r0 = potential_defarg(r0, 1.0);
		auto _alpha = potential_defarg(alpha, 2.0);

        FPTYPE defaultMin;
        if(_r0 > 0) defaultMin = _alpha <= 1 ? _r0 : 0.1 * _r0;
		else defaultMin = 0.1;
        
        return potential_checkerr(potential_create_power(potential_defarg(k, 1.0), 
														 _r0, 
														 _alpha, 
														 potential_defarg(min, defaultMin), 
														 potential_defarg(max, _r0 > 0 ? 3.0 * _r0 : 3.0), 
														 potential_defarg(tol, _alpha >= 1.0 ? 0.001 : 0.01)));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

Potential *TissueForge::Potential::dpd(FPTYPE *alpha, FPTYPE *gamma, FPTYPE *sigma, FPTYPE *cutoff, bool *shifted) {
    TF_Log(LOG_TRACE);

    try {
        return new DPDPotential(potential_defarg(alpha, 1.0), 
								potential_defarg(gamma, 1.0), 
								potential_defarg(sigma, 1.0), 
								potential_defarg(cutoff, 1.0), 
								potential_defarg(shifted, false));
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

util::Differentiator *customDiff = NULL;

FPTYPE customDiffEval_fp(FPTYPE r) {
	return customDiff->fnp(r, 1);
}

FPTYPE customDiffEval_f6p(FPTYPE r) {
	return customDiff->fnp(r, 6);
}

Potential *TissueForge::Potential::custom(FPTYPE min, FPTYPE max, FPTYPE (*f)(FPTYPE), FPTYPE (*fp)(FPTYPE), FPTYPE (*f6p)(FPTYPE), FPTYPE *tol, uint32_t *flags) {
	TF_Log(LOG_TRACE);

    try {
		struct Potential *p = new Potential();

		bool differencing = fp == NULL || f6p == NULL;

		if (differencing) {
			customDiff = new util::Differentiator(f, min, max);
			if (fp == NULL) fp = &customDiffEval_fp;
			if (f6p == NULL) f6p = &customDiffEval_f6p;
		}

		p->name = "Custom";
		p->flags = potential_defarg(flags, POTENTIAL_R);

		int err;

		if ((err = potential_init(p, f, fp, f6p, min, max, potential_defarg(tol, 0.001)))) {
			TF_Log(LOG_ERROR) << "error creating potential: " << potential_err_msg[-err];
			aligned_Free(p);

			if (differencing) {
				delete customDiff;
				customDiff = 0;
			}

			return NULL;
		}

		p->a = min;
		p->b = max;

		if (differencing) {
			delete customDiff;
			customDiff = 0;
		}

		return potential_checkerr(p);
    }
    catch (const std::exception &e) {
        tf_exp(e);
        return NULL;
    }
}

FPTYPE TissueForge::Potential::getMin() {
    return a;
}

FPTYPE TissueForge::Potential::getMax() {
    return b;
}

FPTYPE TissueForge::Potential::getCutoff() {
    return b;
}

std::pair<FPTYPE, FPTYPE> TissueForge::Potential::getDomain() {
    return std::make_pair(a, b);
}

int TissueForge::Potential::getIntervals() {
    return n;
}

bool TissueForge::Potential::getBound() {
    return (bool)flags & POTENTIAL_BOUND;
}

void TissueForge::Potential::setBound(const bool &_bound) {
    if(_bound) flags |= POTENTIAL_BOUND;
    else flags &= ~POTENTIAL_BOUND;
}

FPTYPE TissueForge::Potential::getR0() {
    return r0_plusone - 1.0;
}

void TissueForge::Potential::setR0(const FPTYPE &_r0) {
    r0_plusone = _r0 + 1.0;
}

bool TissueForge::Potential::getShifted() {
    return (bool)flags & POTENTIAL_SHIFTED;
}

bool TissueForge::Potential::getPeriodic() {
	return (bool) flags & POTENTIAL_PERIODIC;
}

bool TissueForge::Potential::getRSquare() {
    return (bool)flags & POTENTIAL_R2;
}

static FPTYPE potential_create_well_k;
static FPTYPE potential_create_well_r0;
static FPTYPE potential_create_well_n;


/* the potential functions */
static FPTYPE potential_create_well_f(FPTYPE r) {
    return potential_create_well_k/Power(-r + potential_create_well_r0,potential_create_well_n);
}

static FPTYPE potential_create_well_dfdr(FPTYPE r) {
    return potential_create_well_k * potential_create_well_n *
            Power(-r + potential_create_well_r0,-1 - potential_create_well_n);
}

static FPTYPE potential_create_well_d6fdr6(FPTYPE r) {
    return -(potential_create_well_k*(-5 - potential_create_well_n)*
            (-4 - potential_create_well_n)*(-3 - potential_create_well_n)*
            (-2 - potential_create_well_n)*(-1 - potential_create_well_n)*
            potential_create_well_n*
            Power(-r + potential_create_well_r0,-6 - potential_create_well_n));
}


Potential *TissueForge::potential_create_well(FPTYPE k, FPTYPE n, FPTYPE r0, FPTYPE tol, FPTYPE min, FPTYPE max)
{
    Potential *p = new Potential();

    p->flags =  POTENTIAL_R2  | POTENTIAL_LJ126 ;
    p->name = "Well";

    /* fill this potential */
    potential_create_well_k = k;
    potential_create_well_r0 = r0;
    potential_create_well_n = n;

    if (potential_init(p,
            &potential_create_well_f,
            &potential_create_well_dfdr,
            &potential_create_well_d6fdr6,
            min, max, tol) < 0) {
        aligned_Free(p);
        return NULL;
    }

    /* return it */
    return p;
}


static FPTYPE potential_create_glj_e;
static FPTYPE potential_create_glj_m;
static FPTYPE potential_create_glj_n;
static FPTYPE potential_create_glj_r0;
static FPTYPE potential_create_glj_k;

/* the potential functions */
static FPTYPE potential_create_glj_f(FPTYPE r) {
    FPTYPE e = potential_create_glj_e;
    FPTYPE n = potential_create_glj_n;
    FPTYPE m = potential_create_glj_m;
    FPTYPE r0 = potential_create_glj_r0;
    FPTYPE k = potential_create_glj_k;
    return k*Power(-r + r0,2) + (e*(-(n*Power(r0/r,m)) + m*Power(r0/r,n)))/(-m + n);
}

static FPTYPE potential_create_glj_dfdr(FPTYPE r) {
    FPTYPE e = potential_create_glj_e;
    FPTYPE n = potential_create_glj_n;
    FPTYPE m = potential_create_glj_m;
    FPTYPE r0 = potential_create_glj_r0;
    FPTYPE k = potential_create_glj_k;
    return 2*k*(-r + r0) + (e*((m*n*r0*Power(r0/r,-1 + m))/Power(r,2) - (m*n*r0*Power(r0/r,-1 + n))/Power(r,2)))/(-m + n);
}

static FPTYPE potential_create_glj_d6fdr6(FPTYPE r) {
    FPTYPE e = potential_create_glj_e;
    FPTYPE n = potential_create_glj_n;
    FPTYPE m = potential_create_glj_m;
    FPTYPE r0 = potential_create_glj_r0;

    return (e*(-(n*(((-5 + m)*(-4 + m)*(-3 + m)*(-2 + m)*(-1 + m)*m*Power(r0,6)*Power(r0/r,-6 + m))/Power(r,12) +
                    (30*(-4 + m)*(-3 + m)*(-2 + m)*(-1 + m)*m*Power(r0,5)*Power(r0/r,-5 + m))/Power(r,11) +
                    (300*(-3 + m)*(-2 + m)*(-1 + m)*m*Power(r0,4)*Power(r0/r,-4 + m))/Power(r,10) +
                    (1200*(-2 + m)*(-1 + m)*m*Power(r0,3)*Power(r0/r,-3 + m))/Power(r,9) +
                    (1800*(-1 + m)*m*Power(r0,2)*Power(r0/r,-2 + m))/Power(r,8) + (720*m*r0*Power(r0/r,-1 + m))/Power(r,7))
                 ) + m*(((-5 + n)*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(r0,6)*Power(r0/r,-6 + n))/Power(r,12) +
                        (30*(-4 + n)*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(r0,5)*Power(r0/r,-5 + n))/Power(r,11) +
                        (300*(-3 + n)*(-2 + n)*(-1 + n)*n*Power(r0,4)*Power(r0/r,-4 + n))/Power(r,10) +
                        (1200*(-2 + n)*(-1 + n)*n*Power(r0,3)*Power(r0/r,-3 + n))/Power(r,9) +
                        (1800*(-1 + n)*n*Power(r0,2)*Power(r0/r,-2 + n))/Power(r,8) + (720*n*r0*Power(r0/r,-1 + n))/Power(r,7))))
    /(-m + n);
}


Potential *TissueForge::potential_create_glj(FPTYPE e, FPTYPE m, FPTYPE n, FPTYPE k,
                                  FPTYPE r0, FPTYPE min, FPTYPE max,
                                  FPTYPE tol, bool shifted)
{
    TF_Log(LOG_DEBUG) << "e: " << e << ", r0: " << r0 << ", m: " << m << ", n:"
                   << n << ", k:" << k << ", min: " << min << ", max: " << max << ", tol: " << tol;
    Potential *p = new Potential();
    
    p->flags =  POTENTIAL_R2  | POTENTIAL_LJ126;
    p->name = "Generalized Lennard-Jones";
    
    /* fill this potential */
    potential_create_glj_e = e;
    potential_create_glj_n = n;
    potential_create_glj_m = m;
    potential_create_glj_r0 = r0;
    potential_create_glj_k = k;
    
    if (potential_init(p,
                       &potential_create_glj_f,
                       &potential_create_glj_dfdr,
                       &potential_create_glj_d6fdr6,
                       min, max, tol) < 0) {
        aligned_Free(p);
        return NULL;
    }

	if(shifted) {
		potential_shift(p, r0);
	} 
	else {
		potential_scale(p);
	}
    
    /* return it */
    return p;
}

static FPTYPE morse_d;
static FPTYPE morse_a;
static FPTYPE morse_r0;

/* the potential functions */
static FPTYPE potential_create_morse_f(FPTYPE r) {
    
    FPTYPE d = morse_d;
    FPTYPE a = morse_a;
	FPTYPE r0 = morse_r0;
    
    return d * (Power(M_E, 2 * a * (r0 - r)) - 2 * Power(M_E, a * (r0 - r)));
}

static FPTYPE potential_create_morse_dfdr(FPTYPE r) {
    
    FPTYPE d = morse_d;
    FPTYPE a = morse_a;
	FPTYPE r0 = morse_r0;
    
    return - 2 * a * d * (Power(M_E, 2 * a * (r0 - r)) - Power(M_E, a * (r0 - r)));
}

static FPTYPE potential_create_morse_d6fdr6(FPTYPE r) {
    
    FPTYPE d = morse_d;
    FPTYPE a = morse_a;
	FPTYPE r0 = morse_r0;
    
    return 2 * Power(a, 6) * d * (32 * Power(M_E, 2 * a * (r0 - r)) - Power(M_E, a * (r0 - r)));
}



static FPTYPE potential_create_morse_shifted_f(FPTYPE r) {
    
    FPTYPE d = morse_d;
    FPTYPE a = morse_a;
    
    return d*(Power(M_E,-2*a*(-1 + r)) - 2/Power(M_E,a*(-1 + r)));
}

static FPTYPE potential_create_morse_shifted_dfdr(FPTYPE r) {
    
    FPTYPE d = morse_d;
    FPTYPE a = morse_a;
    
    return d*((-2*a)/Power(M_E,2*a*(-1 + r)) + (2*a)/Power(M_E,a*(-1 + r)));
}

static FPTYPE potential_create_morse_shifted_d6fdr6(FPTYPE r) {
    
    FPTYPE d = morse_d;
    FPTYPE a = morse_a;
    
    return d*((64*Power(a,6))/Power(M_E,2*a*(-1 + r)) - (2*Power(a,6))/Power(M_E,a*(-1 + r)));
}




Potential *TissueForge::potential_create_morse(FPTYPE d, FPTYPE a, FPTYPE r0,
                                   FPTYPE min, FPTYPE max, FPTYPE tol, bool shifted)
{
    Potential *p = new Potential();
    
    p->flags =  POTENTIAL_R2;
    p->name = "Morse";
    
    /* fill this potential */
    morse_d = d;
    morse_a = a;
	morse_r0 = r0;

	if(shifted) {
		if(potential_init(p, &potential_create_morse_shifted_f, &potential_create_morse_shifted_dfdr, &potential_create_morse_shifted_d6fdr6, min + 1, max + 1, tol) < 0) {
			aligned_Free(p);
			return NULL;
		}
		potential_shift(p, r0);
	} 
	else {
		if(potential_init(p, &potential_create_morse_f, &potential_create_morse_dfdr, &potential_create_morse_d6fdr6, min, max, tol) < 0) {
			aligned_Free(p);
			return NULL;
		}
	}

    /* return it */
    return p;
}


namespace TissueForge::io {


	#define TF_POTENTIALIOTOEASY(fe, key, member) \
		fe = new IOElement(); \
		if(toFile(member, metaData, fe) != S_OK)  \
			return E_FAIL; \
		fe->parent = fileElement; \
		fileElement->children[key] = fe;

	#define TF_POTENTIALIOFROMEASY(feItr, children, metaData, key, member_p) \
		feItr = children.find(key); \
		if(feItr == children.end() || fromFile(*feItr->second, metaData, member_p) != S_OK) \
			return E_FAIL;

	HRESULT toFile(CoulombRPotential *dataElement, const MetaData &metaData, IOElement *fileElement) {

		IOElement *fe;

		TF_POTENTIALIOTOEASY(fe, "modes", dataElement->modes);
		TF_POTENTIALIOTOEASY(fe, "q", dataElement->q);
		TF_POTENTIALIOTOEASY(fe, "kappa", dataElement->kappa);

		fileElement->type = "CoulombRPotential";
		
		return S_OK;
	}

	HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, CoulombRPotential **dataElement) {

		IOChildMap::const_iterator feItr;

		unsigned int modes;
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "modes", &modes);
		FPTYPE q, kappa;
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "q", &q);
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "kappa", &kappa);
		FPTYPE a, b;
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "a", &a);
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "b", &b);

		*dataElement = (CoulombRPotential*)Potential::coulombR(q, kappa, a, b, &modes);

		return S_OK;
	}

	template <>
	HRESULT toFile(Potential *dataElement, const MetaData &metaData, IOElement *fileElement) {

		IOElement *fe;

		fileElement->type = "Potential";

		TF_POTENTIALIOTOEASY(fe, "kind", dataElement->kind);
		TF_POTENTIALIOTOEASY(fe, "flags", dataElement->flags);

		std::string name = dataElement->name;
		TF_POTENTIALIOTOEASY(fe, "name", name);

		// Handle kind
		
		if(dataElement->kind == POTENTIAL_KIND_COMBINATION) {
			if(dataElement->pca != NULL) 
				TF_POTENTIALIOTOEASY(fe, "PotentialA", dataElement->pca);
			if(dataElement->pcb != NULL) 
				TF_POTENTIALIOTOEASY(fe, "PotentialB", dataElement->pcb);
			return S_OK;
		}

		TF_POTENTIALIOTOEASY(fe, "a", dataElement->a);
		TF_POTENTIALIOTOEASY(fe, "b", dataElement->b);

		if(dataElement->kind == POTENTIAL_KIND_DPD) {
			return toFile((DPDPotential*)dataElement, metaData, fileElement);
		} 

		if(dataElement->kind == POTENTIAL_KIND_BYPARTICLES) {
			if(dataElement->flags | POTENTIAL_COULOMBR) {
				return toFile((CoulombRPotential*)dataElement, metaData, fileElement);
			}
		}
		
		// Regular kind

		std::vector<FPTYPE> alpha;
		for(unsigned int i = 0; i < 4; i++) 
			alpha.push_back(dataElement->alpha[i]);
		TF_POTENTIALIOTOEASY(fe, "alpha", alpha);

		TF_POTENTIALIOTOEASY(fe, "n", dataElement->n);

		std::vector<FPTYPE> c;
		for(unsigned int i = 0; i < (dataElement->n + 1) * potential_chunk; i++) {
			c.push_back(dataElement->c[i]);
		}
		TF_POTENTIALIOTOEASY(fe, "c", c);

		TF_POTENTIALIOTOEASY(fe, "r0_plusone", dataElement->r0_plusone);
		TF_POTENTIALIOTOEASY(fe, "mu", dataElement->mu);

		std::vector<FPTYPE> offset;
		for(unsigned int i = 0; i < 3; i++) 
			offset.push_back(dataElement->offset[i]);
		TF_POTENTIALIOTOEASY(fe, "offset", offset);

		return S_OK;
	}

	template <>
	HRESULT fromFile(const IOElement &fileElement, const MetaData &metaData, Potential **dataElement) {

		IOChildMap::const_iterator feItr;

		uint32_t kind;
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "kind", &kind);

		uint32_t flags;
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "flags", &flags);

		if(kind == POTENTIAL_KIND_DPD) {
			DPDPotential *dpdp = NULL;
			if(fromFile(fileElement, metaData, &dpdp) != S_OK) 
				return E_FAIL;
			*dataElement = dpdp;
			return S_OK;
		}
		else if(kind == POTENTIAL_KIND_BYPARTICLES) {
			if(flags | POTENTIAL_COULOMBR) {
				CoulombRPotential *pcr = NULL;
				if(fromFile(fileElement, metaData, &pcr) != S_OK) 
					return E_FAIL;
				*dataElement = pcr;
				return S_OK;
			}
		}

		Potential *p = new Potential();

		p->kind = kind;
		p->flags = flags;

		std::string name;
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "name", &name);
		char *cname = new char[name.size() + 1];
		std::strcpy(cname, name.c_str());
		p->name = cname;
		
		if(p->kind == POTENTIAL_KIND_COMBINATION) {
			if(fileElement.children.find("PotentialA") != fileElement.children.end()) {
				p->pca = NULL;
				TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "PotentialA", &p->pca);
			}
			if(fileElement.children.find("PotentialB") != fileElement.children.end()) {
				p->pcb = NULL;
				TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "PotentialB", &p->pcb);
			}
			*dataElement = p;
			return S_OK;
		} 
		
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "a", &p->a);
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "b", &p->b);

		std::vector<FPTYPE> alpha;
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "alpha", &alpha);
		for(unsigned int i = 0; i < 4; i++) 
			p->alpha[i] = alpha[i];

		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "n", &p->n);
		
		std::vector<FPTYPE> c;
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "c", &c);
		p->c = (FPTYPE*)malloc(sizeof(FPTYPE) * (p->n + 1) * potential_chunk);
		for(unsigned int i = 0; i < (p->n + 1) * potential_chunk; i++) 
			p->c[i] = c[i];

		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "r0_plusone", &p->r0_plusone);
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "mu", &p->mu);

		std::vector<FPTYPE> offset;
		TF_POTENTIALIOFROMEASY(feItr, fileElement.children, metaData, "offset", &offset);
		for(unsigned int i = 0; i < 3; i++) 
			p->offset[i] = offset[i];

		*dataElement = p;

		return S_OK;
	}

};
