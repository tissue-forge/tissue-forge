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

#ifndef _MDCORE_INCLUDE_TF_FPTYPE_H_
#define _MDCORE_INCLUDE_TF_FPTYPE_H_

#include <mdcore_config.h>
    
/* Define some macros for single/double precision vector operations. */
#if defined(FPTYPE_SINGLE)
    #if defined(__AVX__)
        #define VEC_SINGLE
        #define VEC_SIZE 8
        #define VEC_ALIGN 32
        #define VECTORIZE
    #elif ( defined(__SSE__) || defined(__ALTIVEC__) )
        #define VEC_SINGLE
        #define VEC_SIZE 4
        #define VEC_ALIGN 16
        #define VECTORIZE
    #endif
#else
    #if defined(__AVX__)
        #define VEC_DOUBLE
        #define VEC_SIZE 4
        #define VEC_ALIGN 32
        #define VECTORIZE
    #elif defined(__SSE2__)
        #define VEC_DOUBLE
        #define VEC_SIZE 4
        #define VEC_ALIGN 16
        #define VECTORIZE
    #endif
#endif

#undef VECTORIZE

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h> //for Intel __m128i datatype
#include <immintrin.h> //for Intel __m256i datatype
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h> //For ARM uint64x2_t datatype
#endif



/** Macro to easily define vector types. */
#define simd_vector(elcount, type)  __attribute__((vector_size((elcount)*sizeof(type)))) type


namespace TissueForge { 


    /**
     * @brief Inlined function to compute the distance^2 between two vectors.
     *
     * @param x1 The first vector.
     * @param x2 The second vector.
     * @param dx An array in which @c x1 - @c x2 will be stored.
     *
     * @return The Euclidian distance squared between @c x1 and @c x2.
     *
     * Depending on the processor features, this function will use
     * SSE registers and horizontal adds.
     */
    
    TF_ALWAYS_INLINE FPTYPE fptype_r2 ( FPTYPE *x1 , FPTYPE *x2 , FPTYPE *dx ) {

    #if defined(VECTORIZE) && defined(FPTYPE_SINGLE) && defined(__SSE4_1__)
        union {
            simd_vector(4,float) v;
            float f[4];
            } a, b, c, d;
            
        /* Load x1 and x2 into a and b. */
        a.v = _mm_load_ps( x1 );
        b.v = _mm_load_ps( x2 );
        
        /* Compute the difference and store in dx. */
        c.v = a.v - b.v;
        _mm_store_ps( dx , c.v );
        
        /* Use the built-in dot-product instruction. */
        d.v = _mm_dp_ps( c.v , c.v , 0x71 );
        
        /* Return the sum of squares. */
        return d.f[0];
    #elif defined(VECTORIZE) && defined(FPTYPE_SINGLE) && defined(__SSE3__)
        union {
            simd_vector(4,float) v;
            float f[4];
            } a, b, c, d;
            
        /* Load x1 and x2 into a and b. */
        a.v = _mm_load_ps( x1 );
        b.v = _mm_load_ps( x2 );
        
        /* Compute the difference and store in dx. */
        c.v = a.v - b.v;
        _mm_store_ps( dx , c.v );
        
        /* Square the entries (use a different register so that c can be stored). */
        d.v = c.v * c.v;
        
        /* Add horizontally twice to get the sum of the four entries
        in the lowest float. */
        d.v = _mm_hadd_ps( d.v , d.v );
        d.v = _mm_hadd_ps( d.v , d.v );
        
        /* Return the sum of squares. */
        return d.f[0];
    #elif defined(VECTORIZE) && defined(FPTYPE_DOUBLE) && defined(__AVX__)
        union {
            __m256d v;
            double f[4];
            } a, b, c, d;
            
        /* Load x1 and x2 into a and b. */
        a.v = _mm256_load_pd( x1 );
        b.v = _mm256_load_pd( x2 );
        
        /* Compute the difference and store in dx. */
        c.v = a.v - b.v;
        _mm256_store_pd( dx , c.v );
        
        /* Square the entries (use a different register so that c can be stored). */
        d.v = c.v * c.v;
        
        /* Add horizontally twice to get the sum of the four entries
        in the lowest double. */
        d.v = _mm256_hadd_pd( d.v , d.v );
        
        /* Return the sum of squares. */
        return d.f[0] + d.f[2];
    #elif defined(VECTORIZE) && defined(FPTYPE_DOUBLE) && defined(__SSE4_1__)
        union {
            simd_vector(2,double) v;
            double f[2];
            } a1, a2, b1, b2, c1, c2, d1;
            
        /* Load x1 and x2 into a and b. */
        a1.v = _mm_load_pd( x1 );
        b1.v = _mm_load_pd( x2 );
        a2.v = _mm_load_pd( &x1[2] );
        b2.v = _mm_load_pd( &x2[2] );
        
        /* Compute the difference and store in dx. */
        c1.v = a1.v - b1.v;
        c2.v = a2.v - b2.v;
        _mm_store_pd( dx , c1.v );
        _mm_store_pd( &dx[2] , c2.v );
        
        /* Use the built-in dot-product instruction. */
        d1.v = _mm_dp_pd( c1.v , c1.v , 0x31 ) + c2.v * c2.v;
        
        /* Return the sum of squares. */
        return d1.f[0];
    #elif defined(VECTORIZE) && defined(FPTYPE_DOUBLE) && defined(__SSE3__)
        union {
            simd_vector(2,double) v;
            double f[2];
            } a1, a2, b1, b2, c1, c2, d1, d2;
            
        /* Load x1 and x2 into a and b. */
        a1.v = _mm_load_pd( x1 );
        b1.v = _mm_load_pd( x2 );
        a2.v = _mm_load_pd( &x1[2] );
        b2.v = _mm_load_pd( &x2[2] );
        
        /* Compute the difference and store in dx. */
        c1.v = a1.v - b1.v;
        c2.v = a2.v - b2.v;
        _mm_store_pd( dx , c1.v );
        _mm_store_pd( &dx[2] , c2.v );
        
        /* Square the entries (use a different register so that c can be stored). */
        d1.v = c1.v * c1.v;
        d2.v = c2.v * c2.v;
        
        /* Add horizontally twice to get the sum of the four entries
        in the lowest double. */
        d1.v = _mm_hadd_pd( d1.v , d2.v );
        d1.v = _mm_hadd_pd( d1.v , d1.v );
        
        /* Return the sum of squares. */
        return d1.f[0];
    #else
        dx[0] = x1[0] - x2[0];
        dx[1] = x1[1] - x2[1];
        dx[2] = x1[2] - x2[2];
        return dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
    #endif

    }

};

#endif // _MDCORE_INCLUDE_TF_FPTYPE_H_