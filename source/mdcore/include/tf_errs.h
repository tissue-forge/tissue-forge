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
 * @file tf_errs.h
 * 
 */

#ifndef _MDCORE_INCLUDE_TF_ERRS_H_
#define _MDCORE_INCLUDE_TF_ERRS_H_
#include "tf_platform.h"
#include "stdio.h"
#include <string>

/* Some defines. */
#define errs_maxstack                           100

#define errs_err_ok                             0
#define errs_err_io                             -1


namespace TissueForge {


    /* Global variables. */
    CAPI_DATA(int) errs_err;
    CAPI_DATA(const char *) errs_err_msg[];

    /* Functions. */

    CAPI_FUNC(int) errs_register( int id , const char *msg , int line , const char *func , const char *file );
    CAPI_FUNC(int) errs_dump(FILE *out );
    CAPI_FUNC(void) errs_clear();

    std::string errs_getstring(int id);

};

#endif // _MDCORE_INCLUDE_TF_ERRS_H_