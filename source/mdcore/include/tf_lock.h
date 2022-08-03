/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

#ifndef _MDCORE_INCLUDE_TF_LOCK_H_
#define _MDCORE_INCLUDE_TF_LOCK_H_

#include "tf_platform.h"


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
        while ( __sync_val_compare_and_swap( l , 0 , 1 ) != 0 )
            while( *l );
        return 0;
        }
    #define lock_trylock( l ) ( ( *(l) ) ? 1 : __sync_val_compare_and_swap( l , 0 , 1 ) )
    #define lock_unlock( l ) ( __sync_val_compare_and_swap( l , 1 , 0 ) != 1 )
#endif

#endif // _MDCORE_INCLUDE_TF_LOCK_H_