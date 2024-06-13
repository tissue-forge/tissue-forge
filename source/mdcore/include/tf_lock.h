/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

#ifndef _MDCORE_INCLUDE_TF_LOCK_H_
#define _MDCORE_INCLUDE_TF_LOCK_H_

#include "tf_platform.h"

#if (defined(_MSC_VER) && !defined(__GNUC__))
// #include <winnt.h>
#include <intrin.h>
#endif


#if (defined(_MSC_VER) && !defined(__GNUC__))

TF_ALWAYS_INLINE unsigned
InterlockedExchangeAdd(int* Addend, int Value) {
    return (unsigned)_InterlockedExchangeAdd((long*)Addend, (long)Value);
}

#  define sync_val_compare_and_swap(x, y, z) (\
   sizeof *(x) == sizeof(char)    ? _InterlockedCompareExchange8 ((char*)   (x), (char)   (z), (char)   (y)) : \
   sizeof *(x) == sizeof(short)   ? _InterlockedCompareExchange16((short*)  (x), (short)  (z), (short)  (y)) : \
   sizeof *(x) == sizeof(long)    ? _InterlockedCompareExchange  ((long*)   (x), (long)   (z), (long)   (y)) : \
   sizeof *(x) == sizeof(int64_t) ? InterlockedCompareExchange64 ((int64_t*)(x), (int64_t)(z), (int64_t)(y)) : \
                                    (assert(!"Type error in sync_val_compare_and_swap"), 0))
#  define sync_fetch_and_add(a, b) _InterlockedExchangeAdd(a, b)
#  define sync_fetch_and_sub(a, b) _InterlockedExchangeAdd(a, -b)
#else
#  define sync_val_compare_and_swap __sync_val_compare_and_swap
#  define sync_fetch_and_add(a, b) __sync_fetch_and_add(a, b)
#  define sync_fetch_and_sub(a, b) __sync_fetch_and_sub(a, b)
#endif


#ifdef PTHREAD_LOCK
    #define lock_type pthread_spinlock_t
    #define lock_init( l ) ( pthread_spin_init( l , PTHREAD_PROCESS_PRIVATE ) != 0 )
    #define lock_destroy( l ) ( pthread_spin_destroy( l ) != 0 )
    #define lock_lock( l ) ( pthread_spin_lock( l ) != 0 )
    #define lock_trylock( l ) ( pthread_spin_lock( l ) != 0 )
    #define lock_unlock( l ) ( pthread_spin_unlock( l ) != 0 )
#else
    #define lock_type volatile int
    #define lock_init( l ) ( *l = 0 )
    #define lock_destroy( l ) 0
    TF_ALWAYS_INLINE int lock_lock ( volatile int *l ) {
        while ( sync_val_compare_and_swap( l , 0 , 1 ) != 0 )
            while( *l );
        return 0;
        }
    #define lock_trylock( l ) ( ( *(l) ) ? 1 : sync_val_compare_and_swap( l , 0 , 1 ) )
    #define lock_unlock( l ) ( sync_val_compare_and_swap( l , 1 , 0 ) != 1 )
#endif

#endif // _MDCORE_INCLUDE_TF_LOCK_H_