/*******************************************************************************
 * This file is part of Tissue Forge.
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

#pragma SWIG nowarn=312 // Nested union not currently supported
#pragma SWIG nowarn=314 // '<X>' is a python keyword, renaming to '_<X>'
#pragma SWIG nowarn=325 // Nested class not currently supported
#pragma SWIG nowarn=389 // operator[] ignored (consider using %extend)
#pragma SWIG nowarn=401 // Nothing known about base class
#pragma SWIG nowarn=503 // Can't wrap '<X>' unless renamed to a valid identifier.
#pragma SWIG nowarn=506 // Can't wrap varargs with keyword arguments enabled
#pragma SWIG nowarn=509 // Overloaded method <F> effectively ignored, as it is shadowed
#pragma SWIG nowarn=511 // Can't use keyword arguments with overloaded functions
#pragma SWIG nowarn=560 // Unknown Doxygen command: requires

%include "typemaps.i"

// C++ std::string handling
%include "std_string.i"

// C++ std::map handling
%include "std_map.i"
%include "std_unordered_map.i"

// C++ std::set handling
%include "std_set.i"

// C++ std::vector handling
%include "std_vector.i"

// C++ std::list handling
%include "std_list.i"

// C++ std::pair handling
%include "std_pair.i"

%include "stl.i"

%include "stdint.i"
// STL exception handling
%include "exception.i"

%include "cpointer.i"

#define CAPI_DATA(RTYPE) RTYPE
#define CAPI_FUNC(RTYPE) RTYPE
#define CPPAPI_FUNC(...) __VA_ARGS__
#define CAPI_EXPORT
#define CAPI_STRUCT(RTYPE) struct RTYPE

// Lie to SWIG; so long as these aren't passed to the C compiler, no problem
#define __attribute__(x)
#define TF_ALIGNED(RTYPE, VAL) RTYPE

%begin %{
#ifdef _MSC_VER
#define SWIG_PYTHON_INTERPRETER_NO_DEBUG
#include <corecrt.h>
#endif
%}

%{

#define SWIG_FILE_WITH_INIT

#include "TissueForge_private.h"

// todo: A little hacking here; implement a more sustainable cross-platform solution
#ifndef M_PI
    #define M_PI       3.14159265358979323846   // pi
#endif
using FloatP_t = TissueForge::FloatP_t;

%}
